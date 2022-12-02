from Detector import *
from VGG19 import *
from Resnet import *
from Inception import *
from Xception import *
import os


def main():

    print("Welvcome to our demo")
    print("Which model would you like us to demo\n1. VGG-19\n2. ResNet-50\n3. Inception\n4. Xception")
    choice = input()
    match choice:
        case "1":
            model = VGG19('Saved_Models/VGG19_model.h5')
        case "2":
            model = Resnet('Saved_Models/Resnet50.h5')
        case "3":
            model = Inception('Saved_Models/Inception.h5')
        case "4":
            model = Xception('Saved_Models/Xception_model.h5')


    print("Do you want to give a saved video as input or use webcam")
    print("\n1. Webcam\n2. SavedVideo")
    inputchoice = input()
    match inputchoice:
        case "1":
            videoPath = 0
        case "2":
            videoPath = 'video.avi'

    configPath = os.path.join('model_data','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    modelPath = os.path.join('model_data','frozen_inference_graph.pb')
    classesPath = os.path.join('model_data','coco.names')


    detector = Detector(videoPath,configPath,modelPath,classesPath)

    detector.onVideo(model)




if __name__=="__main__":
    main()



