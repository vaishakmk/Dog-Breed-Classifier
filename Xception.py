from keras.models import load_model
import numpy as np
from glob import glob 



class Xception:
    def __init__(self,model_path):
        self.xception_model = load_model(model_path)
        self.name = "Xception"
        self.dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    def extract_Xception(self,tensor):
        from keras.applications.xception import Xception, preprocess_input
        return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def predict_breed (self,img_array):
        # extract the bottle neck features
        bottleneck_feature = self.extract_Xception(img_array) 
        ## get a vector of predicted values
        predicted_vector = self.xception_model.predict(bottleneck_feature) 
        
        ## return the breed
        return self.dog_names[np.argmax(predicted_vector)]

