from keras.models import load_model
import numpy as np
from glob import glob 



class Resnet:
    def __init__(self,model_path):
        self.resnet_model = load_model(model_path)
        self.name = "Resnet"
        self.dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    def extract_Resnet50(self,tensor):
        from keras.applications.resnet import ResNet50, preprocess_input
        return ResNet50(weights='imagenet', pooling="avg", include_top=False).predict(preprocess_input(tensor))

    def predict_breed(self,img_array):
        #extract bottleneck features
        bottleneck_feature = self.extract_Resnet50(img_array)
        print(bottleneck_feature.shape) #returns (1, 2048)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        # bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        print(bottleneck_feature.shape) #returns (1, 1, 1, 1, 2048) - yes a 5D shape, not 4.
        #obtain predicted vector
        predicted_vector = self.resnet_model.predict(bottleneck_feature) #shape error occurs here
        #return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]




