############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: 2D Convolutional Neural Networks
# Lesson: VGG16

# Citation: 
# PEREIRA, V. (2018). Project: 2D CNN, File: Python-DM-CNN-VGG16.py, GitHub repository:<https://github.com/Valdecy/2D-CNN-VGG16>

############################################################################

# Importing Libraries
import os 
import matplotlib.pyplot as plt 
import numpy as np
from keras import layers
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

import cv2 # print(cv2.__version__)
from PIL import Image

from Util.utilities import train_validation_maker, skin_detector_video, validation_predicition, dir_prediction, single_prediction

############################################################################
                 
# Chose Dataset, Train and Test Directories Paths
path_input = 'D:\\SL-Recognition\\Dataset-01-Libras\\original'

# Calling 'train_test_maker' Function
path_output_1, path_output_2 = train_validation_maker(datasetpath = path_input, test_sample_size = 0.2, file_type = 'image')

# If you already have the train and test directories then just point the path.
#path_input    = 'C:\\Users\\Valdecy\\Downloads\\SL-Recognition\\Dataset-02-ASL\\original'
#path_output_1 = 'C:\\Users\\Valdecy\\Downloads\\SL-Recognition\\Dataset-02-ASL\\dataset\\train'
#path_output_2 = 'C:\\Users\\Valdecy\\Downloads\\SL-Recognition\\Dataset-02-ASL\\dataset\\validation'

############################################################################
                 
# Load the VGG Model
vgg_16_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
for layer in vgg_16_model.layers[:-4]: # Freeze the Layers Except the Last 4
    layer.trainable = False
for layer in vgg_16_model.layers: # Check layers' Status
    print(layer, layer.trainable)

# Create the Model
cnn2D = models.Sequential()
cnn2D.add(vgg_16_model) # Add the VGG Model
cnn2D.add(layers.Flatten()) # Add New Layers
cnn2D.add(layers.Dense(units = 1024))
cnn2D.add(Activation('relu'))
cnn2D.add(layers.Dense(units = len(os.listdir(path_output_1))))
cnn2D.add(BatchNormalization())# After a Dense layer
cnn2D.add(Activation('softmax'))
cnn2D.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(lr = 0.0001), metrics = ['accuracy']) # Compile
cnn2D.summary() # Show a Summary of the Model

# Train Set
train_datagen = image.ImageDataGenerator(rescale = 1./255, samplewise_center = True, samplewise_std_normalization = True, shear_range = 0.10, zoom_range = 0.20, horizontal_flip = True, rotation_range = 15, width_shift_range = 0.10, height_shift_range = 0.10, fill_mode = 'constant') 
training_set  = train_datagen.flow_from_directory('dataset/train', target_size = (224, 224), batch_size = 100, class_mode = 'categorical', shuffle = True, seed = 42) 

# Valid Set
valid_datagen  = image.ImageDataGenerator(rescale = 1./255) 
valid_set      = valid_datagen.flow_from_directory ('dataset/validation',  target_size = (224, 224), batch_size = 15, class_mode = 'categorical', shuffle = True, seed = 42) 

# Train the Model
filepath   = '01-CNN-VGG16.hdf5' 
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min') 
callbacks_list = [checkpoint]

history = cnn2D.fit_generator(training_set, steps_per_epoch = int(training_set.samples/training_set.batch_size), epochs = 2, validation_data = valid_set, validation_steps = int(valid_set.samples/valid_set.batch_size), callbacks = callbacks_list)

############################################################################

#Load the Model
cnn2D.load_weights('01-CNN-VGG16.hdf5')
cnn2D.load_weights('01-CNN-VGG16-0.9456.hdf5')

############################################################################

# Validation Set Prediction
prediction_valid = validation_predicition(model = cnn2D, path = path_output_2, batch_size = valid_set.batch_size, image_size_x = 224, image_size_y = 224, rows = 5, cols = 5)

# Directory Prediction
class_list     = os.listdir(path_output_1)
path_directory = 'D:\\SL-Recognition\\Dataset-02-ASL\\01'

prediction_dir = dir_prediction(model = cnn2D, datasetpath = path_directory, directories = class_list, image_size_x = 224, image_size_y = 224, rows = 1, cols = 5)

# Single Prediction
class_list = os.listdir(path_output_1)
path_image = 'D:\\SL-Recognition\\Dataset-02-ASL\\original\\R\\R5 (2).jpg'

prediction_single = single_prediction(model = cnn2D, path = path_image, directories = class_list, image_size_x = 224, image_size_y = 224)

############################################################################

# Acuracy and Loss
gof_train_acc  = history.history['acc'     ]
gof_train_loss = history.history['loss'    ]
gof_valid_acc  = history.history['val_acc' ]
gof_valid_loss = history.history['val_loss'] 
epochs = range(len(gof_train_acc))

# Graph Accuracy
plt.plot(epochs, gof_train_acc, 'b', label = 'Training')
plt.plot(epochs, gof_valid_acc, 'r', label = 'Validation')
plt.title('ACC')
plt.legend()

# Graph Loss
plt.plot(epochs, gof_train_loss, 'b', label = 'Training')
plt.plot(epochs, gof_valid_loss, 'r', label = 'Validation')
plt.title('LOSS')
plt.legend()

############################################################################

# Predict From Camera
cnn2D.load_weights('01-CNN-VGG16.hdf5')
cnn2D.load_weights('01-CNN-VGG16-0.9456.hdf5')

camera       = cv2.VideoCapture(0)
directories  = os.listdir(path_output_1)
frame_width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_detector = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
boolean = True
count = 0
while (boolean == True):
    count = count + 1
    boolean, frame = camera.read()
    if (face_detector == True and boolean == True):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img = frame, pt1 = (x, y), pt2 = (x + w,y + h), color = (0, 0, 0), thickness = -1)
            roi_gray  = gray [y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
    if (count % 1 == 0 and boolean == True):
        im         = skin_detector_video(frame, background = False, polygon = False)
        im         = Image.fromarray(frame, 'RGB')
        im         = im.resize((224, 224))
        test_image = image.img_to_array(im)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis = 0)
        result     = cnn2D.predict_proba(test_image)
        conf_value = float(result[0][np.argmax(max(result))])*100
        conf_value = "{0:.2f}".format(conf_value)
        pred_class = directories[np.argmax(max(result))] + ' (' + conf_value + '%)'
        print('Class Prediction = ' + pred_class)
    if (boolean == True):
        cv2.putText(img = frame, text = pred_class,  org = (int(frame_width/2), int(frame_height - 20)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255),  thickness = 4) 
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1)
    if (key == ord('q') or boolean == False):
        camera.release()
        cv2.destroyAllWindows()
        break

############################################################################