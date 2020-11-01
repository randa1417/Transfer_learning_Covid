from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import random
import os


num = input("Enter: "
            "\n 1: to script Training & show the Accuracy & Show Graph of Accuracy"
            "\n 2: to load the model that saved & make Predictions"
            "\n your number:"
            )

if num == "1":
    ###################### Important layers of CNN ######################
    model = Sequential()

    model.add(Conv2D( 32 , (3,3) , input_shape=(256,256,3) ,activation='relu' ) )
    model.add(MaxPooling2D(pool_size=(2,2) ) )

    model.add(Conv2D( 32 , (3,3) ,activation='relu' ) )
    model.add(MaxPooling2D(pool_size=(2,2) ) )

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten() )
    model.add(Dense(units=128 , activation='relu' ))
    model.add(Dropout(0.2)) # remove some of line between network --> make faster & less over fit --> it's just between [0 - 1]
    model.add(Dense(units=1 ,# 2: it for 2 classes
                    activation='sigmoid'))

    model.compile(optimizer='adam' ,loss= 'binary_crossentropy' , # best is binary for 2 classe -->{0,1}
                  metrics= ['accuracy'] )


    ###################### Generator for Train & validation ######################
    train_gen = ImageDataGenerator(rescale=1./255 ,
                                   shear_range=0.2 ,
                                   zoom_range=0.2 ,
                                   rotation_range=15,
                                   width_shift_range= 0.1,
                                   height_shift_range= 0.1,
                                   horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)


    ###################### Dataset for Train & validation ######################
    trin_set = train_gen.flow_from_directory('Data_covid-19/train' , target_size=(256,256) , batch_size=3 , class_mode='binary')
    val_set = val_gen.flow_from_directory('Data_covid-19/test' , target_size=(256,256) , batch_size=3 , class_mode='binary')



    ###################### Here the machen will be Start training! ######################
    history = model.fit(trin_set ,
                steps_per_epoch=6 , # how num of image will be train
                epochs=19, # num of cycle for per image & to make more accuracy (high epochs)--> epochs=15 or more
                batch_size=4,
                #validation_split= 0.4 , # remove 20% from data & maker faster & less over fit
                validation_steps=3,
                validation_data=val_set)
    #print(model.summary() )
    #train_acc = history.history['accuracy']
    #val_acc = history.history['accuracy']
    #print(train_acc , "\n" , val_acc)
    train_acc = model.evaluate(trin_set,steps=4)
    val_acc = model.evaluate(val_set,steps=4)

    print('\n####### The Accuracy & Loos List is: #######')
    print('\n - For Train')
    print('Train Accuracy: ' , train_acc )

    print('\n - For validation')
    print('validation Accuracy: ', val_acc)

    print('\n\n####### The Accuracy & Loos percentile #######')
    print('\n - For Training data set:')
    print("%s: %.2f%% " % (model.metrics_names[1], train_acc[1] * 100))
    print("Loss:", train_acc[0] )

    print('\n - For validation data set:')
    print("%s: %.2f%% " % (model.metrics_names[1], val_acc[1] * 100))
    print("Loss:", val_acc[0] )

    ###################### Show graph of differance Accuracy between training & validation ######################
    # print(history.history.keys())
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('validation Accuracy')
    plt.xlabel('Training Accuracy')
    plt.legend(loc='best', shadow=True)
    print(plt.show())

    ### Saving Model ###
    model.save('Covid-19_Deep.h5')








if num == "2":
        ###################### Saving Model !? can useing in another project ######################
        new_model = load_model('Covid-19_Deep.h5')
        #print(new_model.summary())

        ###################### Predictions ######################
        imag_path = 'Data_covid-19/val/covid/nejmoa2001191_f4.jpeg'
        img = image.load_img(imag_path, target_size=(256, 256))
        print(plt.imshow(img), plt.show())

        test_img = image.img_to_array(img)
        test_img = np.expand_dims(test_img, axis=0)
        result = new_model.predict(test_img)

        if result[0][0] == 1:
            prediction = '\nThis is Normal  \n'
        else:
            prediction = '\nThis is Covid \n'
        print(prediction)



        test_gen = ImageDataGenerator(rescale=1./255 )

        test_set = test_gen.flow_from_directory('Data_covid-19/test', target_size=(256, 256), batch_size=3, class_mode='binary')
        test_acc = new_model.evaluate(test_set, steps=4)

        print('\n####### The Accuracy & Loos List is: #######')
        print('\n - For Testing data set:')
        print('Testing Accuracy: ', test_acc)

        print('\n\n####### The Accuracy & Loos percentile: #######')
        print('\n - For Testing data set:')
        print("%s: %.2f%% " % (new_model.metrics_names[1], test_acc[1] * 100))
        print("Loss: ",test_acc[0] )













