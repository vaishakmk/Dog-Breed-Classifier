from keras.models import load_model
import numpy as np
from glob import glob 



class VGG19:
    def __init__(self,model_path):
        self.vgg19_model = load_model(model_path)
        self.name = "VGG19"
        self.dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    def extract_VGG19(self,tensor):
        from keras.applications.vgg19 import VGG19, preprocess_input
        return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def predict_breed (self,img_array):
        # extract the bottle neck features
        bottleneck_feature = self.extract_VGG19(img_array) 
        ## get a vector of predicted values
        predicted_vector = self.vgg19_model.predict(bottleneck_feature) 
        
        ## return the breed
        return self.dog_names[np.argmax(predicted_vector)]

