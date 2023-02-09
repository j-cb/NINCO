import csv
import os

class OpenImagesClassReconstructor:
    """Returns the label of a given OpenImages test image, including OpenImage-O samples. Should work at least up to OpenImages v3, but also for many images in later versions.
    """
    
    def  __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'OpenImages_annotations-human_test.csv'), 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            self.test_labels = {fl[0]: fl[2] for fl in reader}
        with open(os.path.join(os.path.dirname(__file__), 'OpenImages_class-descriptions.csv'), 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            self.classes = {fl[0]: fl[1] for fl in reader}
            
            
    def get_openimages_class(self, image_filename):
        image_identifier = image_filename.split('/')[-1].split('.')[-2].split('_')[-1]
        return self.classes.get(self.test_labels.get(image_identifier, 'class unknown'), 'class unknown')
    
                                
    def __call__(self, image_filename):
        return self.get_openimages_class(image_filename)