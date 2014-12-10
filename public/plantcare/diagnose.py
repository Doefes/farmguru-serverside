from skimage import io
import sys
from includes.plantdoctor import PlantDoctor

def post_response(image_path):
        img = io.imread(image_path)
        doctor = PlantDoctor()
        diagnosis = doctor.diagnose(img)
        print "%s" %diagnosis

if __name__ == '__main__':
    image_path = sys.argv[1];
    post_response(image_path)
