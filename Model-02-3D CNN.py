############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Convolutional Neural Networks
# Lesson: CNN 3D

# Citation: 
# PEREIRA, V. (2018). Project: SLR, File: Model-02-3D CNN.py, GitHub repository:<https://github.com/Valdecy/SLR>

############################################################################

# Importing Libraries
import os 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

from Util.utilities import frames_maker, xy_array_maker

############################################################################

frames_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\train'     , padding = False, max_frames = 40)
frames_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\validation', padding = False, max_frames = 40)

path_input    = 'D:\\SL-Recognition\\Dataset-03-Youtube\\original'
path_output_1 = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\train_frames'
path_output_2 = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\validation_frames'

############################################################################

samples_class = 10
samples       = samples_class*len(os.listdir(path_output_1)) # Total Sum of Video Samples per Class
frames        = 40
frame_height  = 64
frame_width   = 64
channels      = 3

X_train, y_train = xy_array_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\train_frames',  samples = samples, frames = frames,  frame_height = frame_height, frame_width = frame_width, channels = channels)

samples_class = 2
samples       = samples_class*len(os.listdir(path_output_1)) 

X_validation, y_validation = xy_array_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset\\validation_frames',  samples = samples, frames = frames,  frame_height = frame_height, frame_width = frame_width, channels = channels)

############################################################################

cnn3D = Sequential()
cnn3D.add(Conv3D(filters = 64, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv1', strides = (1, 1, 1), input_shape = (frames, frame_height, frame_width, channels), data_format = 'channels_last'))
cnn3D.add(MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), padding = 'valid', name = 'pool1'))
cnn3D.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv2', strides = (1, 1, 1)))
cnn3D.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = 'pool2'))
cnn3D.add(Conv3D(filters = 256, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv3a', strides = (1, 1, 1)))
cnn3D.add(Conv3D(filters = 256, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv3b', strides = (1, 1, 1)))
cnn3D.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = 'pool3'))
cnn3D.add(Conv3D(filters = 512, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv4a', strides = (1, 1, 1)))
cnn3D.add(Conv3D(filters = 512, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name = 'conv4b', strides = (1, 1, 1)))
cnn3D.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = 'pool4'))
cnn3D.add(Conv3D(filters = 512, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name ='conv5a', strides = (1, 1, 1)))
cnn3D.add(Conv3D(filters = 512, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same', name ='conv5b', strides = (1, 1, 1)))
cnn3D.add(ZeroPadding3D(padding = (0, 1, 1)))
cnn3D.add(MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = 'pool5'))
cnn3D.add(Flatten()) # only used if the number of frames is fixed.
cnn3D.add(Dense(units = 4096, activation = 'relu', name = 'fc6'))
cnn3D.add(Dropout(rate = 0.50))
cnn3D.add(Dense(units = 4096, activation = 'relu', name = 'fc7'))
cnn3D.add(Dropout(rate = 0.50))
cnn3D.add(Dense(units = len(os.listdir(path_output_1)), activation = 'softmax', name = 'fc8'))
cnn3D.compile(optimizer = SGD(lr = 0.1, decay = 0.000001, momentum = 0.9, nesterov = True), loss = 'categorical_crossentropy', metrics=['accuracy'])
cnn3D.summary()

history = cnn3D.fit(X_train, y_train, batch_size = 400, epochs = 1, verbose = 1, validation_data = (X_validation, y_validation))
score   = cnn3D.evaluate(X_validation, y_validation, batch_size = 400)
