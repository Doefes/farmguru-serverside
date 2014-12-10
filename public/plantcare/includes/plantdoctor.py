# import matplotlib.pyplot for drawing images to the screen
import matplotlib.pyplot as plt
# import numpy for easy multidimensional array manipulation
import numpy as np
# import MySQL module for database access
import MySQLdb as mdb
import os
import copy
from sklearn import neighbors
from sklearn.externals import joblib

# import needed scikit-image modules
from skimage import io
from skimage.color import rgb2hsv, rgb2gray
from scipy import ndimage
from skimage.filter import threshold_otsu as otsu
from skimage.morphology import disk, erosion 

script_root = os.path.dirname(os.path.realpath(__file__));

class PlantDoctor:
    """
    PlantDoctorWho?
    """
    def __init__(self):
        # retrieve images from database
        db_images = self.get_images()

        # create a local dataset to use for diagnosis
        self.dataset = []
        self.images = []
        for url, disease in db_images:
            img_disease = io.imread(script_root+'/'+url)
            self.dataset.append([img_disease, disease])

    def diagnose(self, img):
        leaf_extracted = self.extract_leaf(img)
        disease_extracted = self.extract_disease(img, leaf_extracted)
        diagnosis = self.analyze_disease(disease_extracted)
        return diagnosis

    def extract_leaf(self, img):
        img_gray = rgb2gray(img)

        otsu_mean = otsu(img_gray)
        otsu_img = img_gray < otsu_mean
        self.images.append([otsu_img, "Otsu"])

        eroded_img = ndimage.binary_fill_holes(erosion(otsu_img, disk(1)))

        markers = ndimage.label(eroded_img)[0]

        self.images.append([markers, "Markers"])
        
        markers_copy = markers.astype(float)
        markers_copy[markers_copy == 0] = np.nan
        axis = 1
        u, indices = np.unique(markers_copy, return_inverse=True)
        u = u[np.nanargmax(np.apply_along_axis(np.bincount, axis, indices.reshape(markers_copy.shape),
                                            None, np.max(indices) + 1), axis=axis)]
        u, indices = np.unique(u, return_inverse=True)

        leaf_marker = int(u[np.argmax(np.bincount(indices))])
        img_mask = img

        for row, column in enumerate(markers[:,:]):
            for index, value in enumerate(column):
                if value != leaf_marker:
                    for subindex, value in enumerate(img[row,index,:]):
                        img_mask[row,index,subindex] = 255

        return img_mask

    def extract_disease(self, img, img_extracted):
        self.images.append([img, "Original"])
        #TODO resize image to speed up analysis (determine breaking point for quality)

        self.images.append([img_extracted, "Extracted Leaf"])
        # set diseased pixels to 1, and healthy pixels to 0 (binary representation of regions)
        img_bin = np.where(rgb2hsv(img_extracted) < 0.15, 1.0, 0.0)

        # create a masked image from the binary image above
        img_mask = copy.deepcopy(img)
        for row, column in enumerate(img_bin[:,:,0]):
            for index, value in enumerate(column):
                if value == 0 and np.mean(img_extracted[row, index]) != 255:
                    for subindex, value in enumerate(img[row,index,:]):
                        img_mask[row,index,subindex] = 255

        return img_mask

           
    def analyze_disease(self, img):
        input_hist = []
        img_mask = rgb2hsv(img)
        for i in range(0,3):
            channel_hist = np.histogram(img_mask[:,:,i], range=(0,1))
            channel_hist = list(channel_hist[0])
            for n in channel_hist:
                input_hist.append(n)

        clf = joblib.load('models/hsvmodel.pkl')
        diagnosis = clf.predict(input_hist)[0]

        self.images.append([img_mask, "Disease: %s " % diagnosis])
        #self.show_images()
        return diagnosis


    def get_images(self):
        images = []

        try:
            con = mdb.connect("localhost", "root", "gabber123", "plantcare");
            cur = con.cursor()
            query = ("SELECT url, name FROM images INNER JOIN diseases ON (images.disease_id = diseases.id)")
            cur.execute(query)
            for i in range(cur.rowcount):
                row = cur.fetchone()
                images.append(row)

            cur.close()
            con.close()

            return images

        except mdb.Error, e:
            print "Error %d: %s" % (e.args[0],e.args[1])

    def show_images(self):
        numb_of_imgs = len(self.images)
        fig = plt.figure()
        n = 1
        for image,title in self.images:
            a = fig.add_subplot(1,numb_of_imgs,n)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * numb_of_imgs)
        plt.show()

    def create_knn_model(self, n_neighbors, c_space):
        dataset_hists = []
        dataset_diseases = []
        index = 0
        for image, disease in self.dataset:
            image = rgb2hsv(image)
            image_hist = []
            for i in range(0,3):
                channel_hist = np.histogram(image[:,:,i], range=(0,1))
                channel_hist = list(channel_hist[0])
                for n in channel_hist:
                        image_hist.append(n)
            dataset_hists.append(image_hist)
            dataset_diseases.append(disease)
            index += 1 
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(dataset_hists, dataset_diseases)
        joblib.dump(clf, 'models/%s.pkl') % c_space

