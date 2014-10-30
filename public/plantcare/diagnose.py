# import io module from scikit-image for opening/reading images from filesystem
from skimage import io
import sys
# import our awesome homemade PlantDoctor class
from includes.plantdoctor import PlantDoctor
# import the flask web framework modules

def post_response(image_path):
        # read the image as NumPy array
        print image_path;
        img = io.imread(image_path)
        # create new instance of our awesome PlantDoctor
        doctor = PlantDoctor()
        # diagnose uploaded image
        diagnosis = doctor.diagnose(img)
        print "%s" %diagnosis

# pythonic equivalent of init() function for this script
if __name__ == '__main__':
    image_path = sys.argv[1];
    post_response(image_path)
