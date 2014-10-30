# import matplotlib.pyplot for drawing images to the screen
import matplotlib.pyplot as plt
# import numpy for easy multidimensional array manipulation
import numpy as np
# import MySQL module for database access
import MySQLdb as mdb

# import needed scikit-image modules
from skimage import io, color
import sys, os
from skimage.color import rgb2hsv, rgb2gray
from skimage.measure import structural_similarity as ssim
from skimage.transform import resize
from skimage.filter import gaussian_filter as gf

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
        for url, disease in db_images:
            img_disease = io.imread(script_root+'/'+url)
            self.dataset.append([img_disease, disease])

    def diagnose(self, img):
        # resize image to speed up analysis
        # set diseased pixels to 1, and healthy pixels to 0 (binary representation of regions)
        img_bin = np.where(rgb2hsv(img) < 0.125, 1.0, 0.0)

        # create a masked image from the binary image above
        img_mask = img
        for row, column in enumerate(img_bin[:,:,0]):
            for index, value in enumerate(column):
                if value == 0:
                    for subindex, value in enumerate(img[row,index,:]):
                        img_mask[row,index,subindex] = 0

        """
            create a SSIM dictionary by comparing the uploaded image with our local dataset.
            SSIM documentation: http://en.wikipedia.org/wiki/Structural_similarity
        """
        ssim_list = {}
        img_mask = rgb2gray(gf(img_mask, sigma=1))
        for image, disease in self.dataset:
            image = rgb2gray(gf(image, sigma=1))
            mask_shape = img_mask.shape
            disease_shape = image.shape
            if(mask_shape != disease_shape):
                img_mask = resize(img_mask, (disease_shape[0], disease_shape[1]))

            ssim_disease = ssim(img_mask, image)
            if disease in ssim_list:
                ssim_list[disease].append(ssim_disease*100)
            else:
                ssim_list[disease] = [ssim_disease*100]

        # determine which dataset (disease) best matches the uploaded image
        highest_similarity = ['disease', 0];
        for disease, value in ssim_list.items():
            for match in value:
                if (match > highest_similarity[1]):
                    highest_similarity = [disease, match]

        # return the name of the diagnosed disease
        return str(highest_similarity[0])

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

    def show_images(self,images,titles=None):
        numb_of_imgs = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1,numb_of_imgs + 1)]
        fig = plt.figure()
        n = 1
        for image,title in zip(images,titles):
            a = fig.add_subplot(1,numb_of_imgs,n)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * numb_of_imgs)
        plt.show()
