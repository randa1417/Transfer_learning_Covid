from tensorflow.keras.applications.resnet import ResNet50 ,ResNet101, preprocess_input , decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import random
import cv2


###################### load_model ######################
new_model = load_model('Covid-19MobilNetV2.h5')


###################### Predictions ######################
img_path = glob('Data_covid-19/test_covid19/covid/*.*')

for path in img_path:
    im = cv2.imread(path)
    im = cv2.resize(im, (256,256))/255
    im = im.reshape(1,256,256,3)
    # print(im.shape)
    # [.8 , .2] ***
    predic = new_model.predict(im)
    #print(predic)
    print(predic.argmax())