from glob import glob

from tensorflow.keras.applications.resnet import ResNet50 ,ResNet101, preprocess_input , decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import tensorflow as tf

import os

from tensorflow.keras import Model


ResNet_50 = ResNet50(input_shape=[256,256,3], weights='imagenet' , include_top=False)

for layer in ResNet_50.layers:
    layer.trainable = False


x = Flatten()(ResNet_50.output)
pred = Dense(2 , activation='sigmoid')(x)
model = Model(inputs=ResNet_50.input , outputs= pred)

#model.summary()
optimizer = tf.keras.optimizers.Adam(lr=.0001, clipnorm=0.0001)

########################## Image DataGenerator ##########################
model.compile(optimizer=optimizer ,loss= 'binary_crossentropy' ,
                  metrics= ['accuracy'] )

train_gen = ImageDataGenerator(rescale=1./255 ,
                                   zoom_range=0.2 ,
                                   width_shift_range= 0.1,
                                   height_shift_range= 0.1,
                               validation_split=0.2)

# val_gen = ImageDataGenerator(rescale=1./255)


trin_set = train_gen.flow_from_directory('Data_covid-19/train', target_size=(256,256), batch_size=16, subset='training')
val_set = train_gen.flow_from_directory('Data_covid-19/train', target_size=(256,256), batch_size=16, subset='validation')

########################## Fit ##########################
history = model.fit(trin_set ,
                steps_per_epoch=60, # how num of image will be train
                epochs=40, # num of cycle for per image & to make more accuracy (high epochs)--> epochs=15 or more
                #validation_split= 0.4 , # remove 20% from data & maker faster & less over fit
                validation_steps=10,
                validation_data=val_set)

# print(trin_set.class_indices)

train_acc = model.evaluate(trin_set,steps=4)
val_acc = model.evaluate(val_set,steps=4)

print('\n\n####### The Accuracy & Loos percentile #######')
print('\n - For Training data set:')
print("%s: %.2f%% " % (model.metrics_names[1], train_acc[1] * 100))
print("Loss:", train_acc[0] )

print('\n - For validation data set:')
print("%s: %.2f%% " % (model.metrics_names[1], val_acc[1] * 100))
print("Loss:", val_acc[0] )

#model.save('Covid-19ResNet50.h5')

'''
 - For Training data set:
    accuracy: 100.00% 
    Loss: 0.04819430410861969

 - For validation data set:
    accuracy: 100.00% 
    Loss: 2.001482926061726e-06

'''


