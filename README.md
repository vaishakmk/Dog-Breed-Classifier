# Dog-Breed-Classifier


Team Members :

    ● Rohith Puvvala Subramanyam
    ● Sai Varshith Talluri
    ● Teja Ramisetty
    ● Vaishak Melarcode Kallampad
    
    
Requirements :

    ● Tensorflow - 2.10.0
    ● OpenCV - 4.6.0
    ● Any IDE supporting python - VS Code, Pycharm
    ● Jupyter Notebook
    ● Keras - 2.10.0
    ● Matplotlib - 3.5.1
    ● Numpy - 1.21.5
    ● Tqdm - 4.64.0
    ● Scikit-learn - 1.0.2


Dataset :

    ● You can download the dog dataset here. Unzip the folder and place it in your working directory
    where all the other python files(.py) and notebooks(.ipynb) exist.
    ● This data set has 8,351 total images with 133 different breeds. The number of available images
    for the models to learn from is about 62 per breed, which might not be enough for CNN.


Files :

    ● dogImages - A folder containing the images separated into training , validation and test folders.
    The link to download is https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip .
    ● bottleneck_features - A folder containing the bottleneck features of the 4 models. Download links
    for all 4 models have been provided in the below section.
    ● model_data - A folder containing the Object detection model. Refer
    https://github.com/haroonshakeel/real_time_object_detection_cpu for more information.
    ● Saved_Models - A folder that contains all the saved models and weights.
    ● outpout - A folder which saves the
    ● All_Model.ipynb - A jupyter notebook for training all the models and saving them.
    ● VGG19.py - contains functions for loading VGG19 model and making predictions.
    ● Resnet.py - contains functions for loading Resnet50 model and making predictions.
    ● Inception.py - contains functions for loading InceptionV3 model and making predictions.
    ● Xception.py - contains functions for loading Xception model and making predictions.
    ● Detector.py - contains object detection model for detecting dogs, creating bounding boxes and
    calling the chosen model for predicting the breed.
    ● main.py - This is the main file which needs to be run. All other files are imported into this file.
    ● video.avi - This is the video of dogs we give as input to our program.


Training :

The training of the different models is performed in the All_Models.ipynb file. There are a total of 5 models
trained in this python notebook.

    ● The “CNN model from scratch” can be trained by executing all the code blocks under it in the
    notebook.
    ● For the rest of the pretrained models we need to download the bottleneck features. Bottleneck
    features are the last activation maps before the fully-connected layers in a pretrained model. We
    remove the last fully connected layer from the model and plug in your layers there.
    ● You can download the bottleneck features for the different models from the below links:
        - VGG-19 bottleneck features : https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz
        - ResNet-50 bottleneck features : https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz
        - Inception bottleneck features : https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz
        - Xception bottleneck features : https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz
    ● Place the downloaded files in a folder bottleneck_features in your working directory.
    ● Once you run all the code blocks in you will have all the models trained on the dog dataset and
    saved in a folder in your directory.

Predicting Dog Breeds - Program Execution :

    ● Run the file main.py.
    ● It will ask for a choice of model among the 4 mentioned above. Give choice.
    ● Then you will get a prompt to choose the input - from a webcam or saved video.
    ● If selecting a live video (webcam), show some images of dogs to the camera and see the output.
    ● If selecting saved video, a previously saved video is passed as input and the model tries
    detecting the dog breeds in that video and saves the output as a video file.
    Results :
    ● The individual accuracies of each model on the test data can be viewed in the All_Model.ipynb
    file.
    ● The real time performance of the models can be tested by running the main file. It passes
    ‘video.mp4’ as input and saves the output video in the output folder. The output video shows a
    bounding box around the dog if detected. The actual breed name is displayed at the bottom in
    green and the predicted breed is displayed in red on top of the bounding box.
