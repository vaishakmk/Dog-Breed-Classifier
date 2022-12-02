import cv2
import numpy as np
import time
from Xception import *
from keras.utils import img_to_array  


class Detector:
    def __init__(self,videoPath,configPath,modelPath,classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
    
        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
    
    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0,'__Background__')
        print(self.classesList)

    def onVideo(self,_model):
        cap = cv2.VideoCapture(self.videoPath)
        model = _model
        if(cap.isOpened()==False):
            print("error opening file")
            return
        
        success ,image = cap.read()
        copy_image = image
        

        # image = cv2.imread('dogImages/test/097.Lakeland_terrier/Lakeland_terrier_06528.jpg')
        out = cv2.VideoWriter(f'output/{model.name}_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 0.5 , (224,224))

        starttime = 0
        while success :
            # image = cv2.resize(image, (224,224))
            currenttime = time.time()
            fps = 1/(currenttime-starttime)
            starttime = currenttime
            classLabelIDs , confidences, bboxs = self.net.detect(image,confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=0.5,nms_threshold=0)
            # cv2.putText(image, str(int(fps)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),2) 

            if len(bboxIdx)!=0:
                for i in range(0,len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]

                    if classLabel=='dog':
                        x,y,w,h = bbox
                        pad =15
                        crop_image = image[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
                        crop_image = cv2.resize(crop_image, (224,224))
                        crop_image_arr = img_to_array(crop_image)
                        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
                        crop_image_arr = np.expand_dims(crop_image_arr, axis=0)
                        prediction = model.predict_breed(crop_image_arr)
                        cv2.rectangle(image,(x,y),(x+w,y+h),color=(255,255,255),thickness=1)
                        cv2.putText(image, prediction, (x, y+15),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),2) 
            out.write(image) 
            cv2.imshow("Result",image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
                        

            success ,image = cap.read()
        cv2.destroyAllWindows( )





