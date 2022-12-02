from keras.models import load_model
import numpy as np
from glob import glob 



class Inception:
    def __init__(self,model_path):
        self.name = "Inception"
        self.inception_model = load_model(model_path)
        self.dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    def extract_InceptionV3(self,tensor):
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
        return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def predict_breed (self,img_array):
        # extract the bottle neck features
        bottleneck_feature = self.extract_InceptionV3(img_array) 
        ## get a vector of predicted values
        predicted_vector = self.inception_model.predict(bottleneck_feature) 
        
        ## return the breed
        return self.dog_names[np.argmax(predicted_vector)]

