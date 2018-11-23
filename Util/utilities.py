############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Convolutional Neural Networks
# Lesson: 2D & 3D

# Citation: 
# PEREIRA, V. (2018). Project: SLR, File: Model-01-2D CNN-VGG16.py, GitHub repository:<https://github.com/Valdecy/SLR>

############################################################################

# Importing Libraries
import os
import re
import shutil
from os import listdir
from os.path import isfile, join
import time
import matplotlib.pyplot as plt 
import numpy as np
#import pandas as pd
from random import randint
from keras.preprocessing import image

import cv2 # print(cv2.__version__)

############################################################################

# Function: Count the Exact Number of Frames from a Video
def count_exact_frames(path_input):
    camera = cv2.VideoCapture(path_input)
    count  = 1
    boolean, img = camera.read()
    while (boolean == True):
      #print('Frame = ' + str(count))
      boolean, img = camera.read()
      count = count + 1
    return count - 1

# Usage
#count_exact_frames(path_input = 'Video-05.mp4')
    
############################################################################

# Function: Transform Video to Frame
def video_to_frames(path_input, path_output = 'none', verbose = 1, max_frames = -1,  height = 'none', width = 'none'):
    camera = cv2.VideoCapture(path_input)
    count  = 1
    file   = os.path.splitext(os.path.basename(path_input))[0]
    boolean, img = camera.read()
    if (height != 'none' and width != 'none'):
        img = cv2.resize(img, (int(height), int(width)))
    while (boolean == True):
      if (verbose == 1):      
          print('Frame = ' + str(count))
      if (path_output == 'none'):
          cv2.imwrite(file + '_frame' + '_%d_.jpg' % count, img) # save frame as JPG file
      else:
          cv2.imwrite(path_output + '\\' + file + '_frame' + '_%d_.jpg' % count, img) # save frame as JPG file
      boolean, img = camera.read()
      count = count + 1
      if(max_frames > 0 and max_frames == count - 1):
          boolean = False
    return count - 1 # necessary to inform total for 'video_frames_padding' function

# Usage
#video_to_frames(path_input = 'Video-05.mp4', verbose = 1, max_frames = -1,  height = 'none', width = 'none')

############################################################################

# Function: Transform Frames to Video
def frames_to_video(path_input, video_name = 'video', fps = 30, extension = '.jpg', show_video = False):
    imgs        = []
    path_output = path_input + '\\' + video_name +'.mp4'
    for f in os.listdir(path_input):
        if f.endswith(extension):
            imgs.append(f)
    image_path = os.path.join(path_input, imgs[0])
    frame      = cv2.imread(image_path)
    cv2.imshow('video', frame)
    frame_height, frame_width, channels = frame.shape
    output = cv2.VideoWriter(filename = path_output, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), fps = 20.0, frameSize = (frame_width, frame_height))
    for img in imgs:
        image_path = os.path.join(path_input, img)
        print('Joining = ' + img)
        frame = cv2.imread(image_path)
        output.write(frame)
        if (show_video == True):
            cv2.imshow('video', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                output.release()
                cv2.destroyAllWindows()
                break
    return

# Usage
#frames_to_video(path_input = 'D:\\SL-Recognition\\Dataset-02-ASL\\01', video_name = '00', fps = 30, extension = '.jpg', show_video = True)

############################################################################

# Function: Padding Video Frames 
def video_frames_padding(path_input, path_output = 'none', lenght = 70, verbose = 1, max_frames = -1):
    total        = video_to_frames(path_input, path_output = path_output, verbose = verbose, max_frames = max_frames)
    file         = os.path.splitext(os.path.basename(path_input))[0]
    camera       = cv2.VideoCapture(path_input)
    frame_width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img          = np.zeros((frame_height, frame_width, 3), np.uint8)
    #img[:]       = (0, 0, 0)
    while (total < lenght):
        total = total + 1
        if (path_output == 'none'):
            cv2.imwrite(file + '_frame_padding' + '_%d_.jpg' % total, img)
        else:
            cv2.imwrite(path_output + '\\' + file + '_frame_padding' + '_%d_.jpg' % total, img)
        if (verbose == 1):
            print('Padding = ' + str(total))
    return

# Usage
#video_frames_padding(path_input = 'Video-05.mp4', lenght = 100, verbose = 1)

############################################################################
    
# Function: Tranform Videos to Frames
def frames_maker(datasetpath, padding = False, max_frames = -1):
    directories = os.listdir(datasetpath)
    up_path     = os.path.dirname(datasetpath)
    name        = os.path.basename(os.path.normpath(datasetpath))
    name        = name + '_frames'
    dataset_dir = mkdir(up_path    , name)
    #dataset_dir = mkdir(up_path    , 'dataset_frames')
    # Copy the Dataset Directory Structure
    start_time = time.time()
    for dirpath, dirnames, filenames in os.walk(datasetpath):
        structure = dataset_dir + dirpath[len(datasetpath):]
        if not os.path.isdir(structure):
            os.mkdir(structure)
    print("Directory Structure Ready!")
    # Max Padding Lenght
    if (padding == True):
        min_lenght = 999999
        max_lenght = 0
        for d in range(0, len(directories)):
            if os.path.isdir(datasetpath + '\\' + directories[d]):
                directory =  datasetpath + '\\' + directories[d]
                with os.scandir(directory) as listOfEntries:  
                    for entry in listOfEntries:
                        if entry.is_file():
                            f = str(entry)
                            f = f.replace('<DirEntry ','')
                            f = f.replace('>','')
                            f = f.replace("'",'')
                            size = count_exact_frames(path_input = datasetpath + '\\' + directories[d] + '\\' + f)
                            if (size > max_lenght):
                                max_lenght = size
                            if (size < min_lenght):
                                min_lenght = size
        print('Frames Length = (' + 'min = ' + str(min_lenght) + ', ' + 'max = ' + str(max_lenght) + ')')
    # Transforming to Frames
    for d in range(0, len(directories)):
        if os.path.isdir(datasetpath + '\\' + directories[d]):
            directory =  datasetpath + '\\' + directories[d]
            print('Working on Directory = ',  directories[d])
            with os.scandir(directory) as listOfEntries:  
                for entry in listOfEntries:
                    if entry.is_file():
                        f = str(entry)
                        f = f.replace('<DirEntry ','')
                        f = f.replace('>','')
                        f = f.replace("'",'')
                        if (padding == False):
                            video_to_frames(path_input = datasetpath + '\\' + directories[d] + '\\' + f, path_output = dataset_dir + '\\' + directories[d], verbose = 0, max_frames = max_frames)
                        else:
                            video_frames_padding(path_input = datasetpath + '\\' + directories[d] + '\\' + f, path_output = dataset_dir + '\\' + directories[d], lenght = max_lenght, verbose = 0, max_frames = max_frames)
    elapsed_time = time.time() - start_time
    print('Done! Total Time = ' + time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
    return

# Usage
#frames_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\original_', padding = False, max_frames = 40)
    
############################################################################
    
# Function: Tranform Images to [X, y] Arrays
def xy_array_maker(datasetpath, samples, frames, frame_height, frame_width, channels):
    directories = os.listdir(datasetpath)
    X = np.zeros([samples, frames, frame_height, frame_width, channels])
    y = np.zeros([samples, len(directories)])
    video_label = -1  # sample label
    
    # Function: List Natural Sorting
    def sorted_aphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key = alphanum_key)
    
    # Copy the Dataset Directory Structure
    start_time = time.time()
    # Transforming to Frames
    for d in range(0, len(directories)):
        f_list = [ ]
        count  =  0
        if os.path.isdir(datasetpath + '\\' + directories[d]):
            directory =  datasetpath + '\\' + directories[d]
            print('Working on Directory = ',  directories[d])
            with os.scandir(directory) as listOfEntries:  
                for entry in listOfEntries:
                    if entry.is_file():
                        f = str(entry)
                        f = f.replace('<DirEntry ','')
                        f = f.replace('>','')
                        f = f.replace("'",'')
                        f_list.append(f)
                f_list = sorted_aphanumeric(f_list)
                if ( len(f_list)/frames > int( (samples/len(directories)) ) ):
                    f_list = f_list[0:int((samples/len(directories))*frames)]
                for file in f_list:
                    if (count % frames == 0):
                        #print(file + str(count) + ' / ' + str(frames) + ' = ' + str(count/frames))
                        video_label = video_label + 1
                        y[video_label, d] = 1
                        count = 0
                    path_file = datasetpath + '\\' + directories[d] + '\\' + file
                    img = cv2.imread(path_file)
                    img = cv2.resize(img, (frame_height, frame_width))
                    img = image.img_to_array(img)
                    X[video_label, count, :, :, :] = img/255.0
                    count = count + 1
    elapsed_time = time.time() - start_time
    print('Done! Total Time = ' + time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
    return X, y

# Usage
#xy_array_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-03-Youtube\\dataset_frames',  samples = 20, frames = 40,  frame_height = 64, frame_width = 64, channels = 3)

############################################################################
    
# Function: Convert Video to mp4    
def convert_to_mp4(path_input, output_name = 'new_video', fps = 30):
    camera = cv2.VideoCapture(path_input)
    frame_width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = output_name + '.mp4'
    output = cv2.VideoWriter(filename = video_name, fourcc = cv2.VideoWriter_fourcc(*'MP4V'), fps = fps, frameSize = (frame_width, frame_height))
    boolean = True
    while (boolean == True):
        boolean, frame = camera.read()
        output.write(frame)
        print('Capturing Frame = ' + str(boolean))
    print('Video Converted!')
    return

# Usage    
#convert_to_mp4(path_input = 'Video-07.mov', output_name = 'Video-07', fps = 30)
    
############################################################################
  
# Function: Skin Detector - Images
def skin_detector_image(path_file, show_image = False, background = False, polygon = False):
    image           = cv2.imread(path_file)
    im_ycrcb        = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    skin_ycrcb_min  = np.array((0,   133,  77), np.uint8)
    skin_ycrcb_max  = np.array((255, 173, 127), np.uint8)
    skin_ycrcb      = cv2.inRange(im_ycrcb, skin_ycrcb_min, skin_ycrcb_max) # binary image
    _, contours, _  = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (polygon == True):
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(image, contours, i, (0, 255, 0), 3)
    if (background == False):
        stencil = np.zeros(image.shape).astype(image.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result = cv2.bitwise_and(image, stencil)
    else:
        result = image
    if (show_image == True):
        cv2.imshow('Skin_Detector', result)
    return result

# Usage
#img = skin_detector_image('D:\\SL-Recognition\\Dataset-02-ASL\\00\\IMG-20181005-WA0008.jpg', show_image = True, background = False, polygon = True)
#cv2.imwrite('D:\\SL-Recognition\\Dataset-02-ASL\\IMG-20181005-WA0010.jpg', img) # Save Image

############################################################################
    
# Function: Skin Detector - Video
def skin_detector_video(frame, background = True, polygon = False):
    im_ycrcb        = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skin_ycrcb_min  = np.array((0,   133,  77), np.uint8)
    skin_ycrcb_max  = np.array((255, 173, 127), np.uint8)
    skin_ycrcb      = cv2.inRange(im_ycrcb, skin_ycrcb_min, skin_ycrcb_max) # binary image
    _, contours, _  = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (polygon == True):
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(frame, contours, i, (0, 255, 0), 3)
    if (background == False):
        stencil = np.zeros(frame.shape).astype(frame.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result = cv2.bitwise_and(frame, stencil)
    else:
        result = frame
    return result

# Usage
#camera  = cv2.VideoCapture(0) # WebCam
#camera  = cv2.VideoCapture('Video-05.mp4') # Video File
#boolean = True
#while (boolean == True):
        #boolean, frame = camera.read() # returns: '_' boolean (cam is opened) and 'frame' (image from camera) 
        #frame = skin_detector_video(frame, background = False, polygon = True)
        #cv2.imshow('Video', frame)
        #key = cv2.waitKey(1)
        #if (key == ord('q') or boolean == False):
            #camera.release()
            #cv2.destroyAllWindows()
            #break

############################################################################
            
# Function: Make Directory
def mkdir(datasetpath, dir_name):
    path = os.path.join(datasetpath, dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Usage
#mkdir(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL', 'new_folder')
    
############################################################################
    
# Function: Train & Validation Copy and Separator
def train_validation_maker(datasetpath, test_sample_size = 0.2, background = True, file_type = 'image'):
    directories = os.listdir(datasetpath)
    up_path     = os.path.dirname(datasetpath)
    dataset_dir = mkdir(up_path    , 'dataset')
    validpath   = mkdir(dataset_dir, 'validation')
    trainpath   = mkdir(dataset_dir, 'train')
    # Copy the Dataset Directory Structure to Train and Test Directories
    start_time = time.time()
    for dirpath, dirnames, filenames in os.walk(datasetpath):
        structure_1 = trainpath + dirpath[len(datasetpath):]
        structure_2 = validpath + dirpath[len(datasetpath):]
        if not os.path.isdir(structure_1):
            os.mkdir(structure_1)
            os.mkdir(structure_2)
    print("Directory Structure Ready!")
    # Copy and Separate Dataset Files in Train and Test Samples
    for d in range(0, len(directories)):
        if os.path.isdir(datasetpath + '\\' + directories[d]):
            directory =  datasetpath + '\\' + directories[d]
            print('Working on Directory = ',  directories[d])
            with os.scandir(directory) as listOfEntries:  
                for entry in listOfEntries:
                    rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                    if entry.is_file():
                        f = str(entry)
                        f = f.replace('<DirEntry ','')
                        f = f.replace('>','')
                        f = f.replace("'",'')
                        if rand > test_sample_size:
                            shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, trainpath + '\\' + directories[d] + '\\' + f)
                            if (file_type == 'image'):
                                image = skin_detector_image(trainpath + '\\' + directories[d] + '\\' + f, show_image = False, background = background, polygon = False)
                                cv2.imwrite(trainpath + '\\' + directories[d] + '\\' + f, image)
                        else:
                            shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, validpath  + '\\' + directories[d] + '\\' + f)
                            if (file_type == 'image'):
                                image = skin_detector_image(validpath + '\\' + directories[d] + '\\' + f, show_image = False, background = background, polygon = False)
                                cv2.imwrite(validpath + '\\' + directories[d] + '\\' + f, image)
    elapsed_time = time.time() - start_time
    print('Done! Total Time = ' + time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
    return trainpath, validpath

# Usage
#path_output_1, path_output_2 = train_validation_maker(datasetpath = 'D:\\SL-Recognition\\Dataset-01-Libras\\original', test_sample_size = 0.2, background = False)

############################################################################

# Function: Directory Cleaner
def dir_cleaner(path):
    directories  = os.listdir(path)
    # Copy and Separate Dataset Files in Train and Test Samples
    for d in range(0, len(directories)):
        if os.path.isdir(path + '\\' + directories[d]):
            directory =  path + '\\' + directories[d]
            print('Cleaning Directory = ', directories[d])
            with os.scandir(directory) as listOfEntries:  
                for entry in listOfEntries:
                    if entry.is_file():
                        f = str(entry)
                        f = f.replace('<DirEntry ','')
                        f = f.replace('>','')
                        f = f.replace("'",'')
                        os.unlink(path + '\\' + directories[d] + '\\' + f)
    print('Done!')
    return

# Usage
#dir_cleaner(path = 'D:\\SL-Recognition\\Dataset-02-ASL\\dataset\\train')
    
############################################################################

# Function: Data Augmentation   
def data_augmentation(datasetpath, flip_h = False, flip_v = False, flip_hv = False, gaussian_noise = False, crop = False, translation = False, sharpen = False, emboss = False, change_color = 'none'):
    directories = os.listdir(datasetpath)
    start_time = time.time()
    for d in range(0, len(directories)):
        if os.path.isdir(datasetpath + '\\' + directories[d]):
            directory =  datasetpath + '\\' + directories[d]
            print('Augmenting Data on Directory = ',  directories[d])
            onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))] 
            for entry in onlyfiles:
                f = str(entry)
                #f = f.replace('<DirEntry ','')
                #f = f.replace('>','')
                f = f.replace("'",'')
                if (flip_h == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'flipped_h_' + f)
                    image = cv2.imread(datasetpath + '\\' + directories[d] + '\\'  + 'flipped_h_' + f)
                    image = cv2.flip(image, 0)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'flipped_h_' + f, image)
                if (flip_v == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'flipped_v_' + f)
                    image = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'flipped_v_' + f)
                    image = cv2.flip(image, 1)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'flipped_v_' + f, image)
                if (flip_hv == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'flipped_hv_' + f)
                    image = cv2.imread(datasetpath + '\\' + directories[d] + '\\'  + 'flipped_hv_' + f)
                    image = cv2.flip(image, -1)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'flipped_hv_' + f, image)
                if (gaussian_noise == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'g_noise_' + f)
                    image   = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'g_noise_' + f)
                    g_noise = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'g_noise_' + f)
                    mean    = ( 100,  100,  100) 
                    sigma   = ( 200,  200,  200)
                    cv2.randn(g_noise, mean, sigma)
                    image = g_noise + image
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'g_noise_' + f, image)
                if (crop == True):
                    shutil.copyfile(datasetpath  + '\\' + directories[d] + '\\' + f, datasetpath  + '\\' + directories[d] + '\\' + 'crop_' + f)  
                    image   = cv2.imread(datasetpath  + '\\' + directories[d] + '\\'  + 'crop_' + f)
                    y1 = randint(0, int(image.shape[0]/2))
                    x1 = randint(0, int(image.shape[1]/2))
                    y2 = randint(int(image.shape[0]/2), image.shape[0])
                    x2 = randint(int(image.shape[1]/2), image.shape[1])
                    image = image[y1:y2, x1:x2]
                    crop_img = image[y1:y2, x1:x2]
                    height, width, depth = image.shape
                    image = cv2.resize(crop_img, (height, width))
                    cv2.imwrite(datasetpath  + '\\' + directories[d] + '\\' + directories[d] + '\\' + 'crop_' + f, image)
                if (translation == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'translated_' + f)  
                    image   = cv2.imread(datasetpath + '\\' + directories[d] + '\\'  + 'translated_' + f)
                    x = randint(-int(image.shape[0]/3), int(image.shape[0]/3))
                    y = randint(-int(image.shape[1]/3), int(image.shape[1]/3))
                    M = np.float32([[1, 0, x],[0, 1, y]])
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) 
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'translated_' + f, image)
                if (sharpen == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\'  + directories[d] + '\\' + 'sharpen_' + f)  
                    image   = cv2.imread(datasetpath + '\\' + directories[d] + '\\'  + 'sharpen_' + f)
                    kernel_sharpening = np.array([[-5, -5, -5],  [-5, 45, -5], [-5, -5, -5]])
                    image = cv2.filter2D(image, -1, kernel_sharpening)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'sharpen_' + f, image)
                if (emboss == True):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\'  + directories[d] + '\\' + 'emboss_' + f)  
                    image   = cv2.imread(datasetpath + '\\' + directories[d] + '\\'  + 'emboss_' + f)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                    kernel_embossing = np.array([[0, -1, -1],  [1, 0, -1], [1, 1, 0]])
                    image = cv2.filter2D(image, -1, kernel_embossing)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'emboss_' + f, image)
                if (change_color == 'yuv' or change_color == 'all'):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'yuv_' + f)
                    image     = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'yuv_' + f) 
                    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'yuv_' + f, yuv_image)
                if((change_color == 'hsv' or change_color == 'all')):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'hsv_' + f)
                    image     = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'hsv_' + f) 
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'hsv_' + f, hsv_image)
                if((change_color == 'ycrcb' or change_color == 'all')):
                    shutil.copyfile(datasetpath + '\\' + directories[d] + '\\' + f, datasetpath + '\\' + directories[d] + '\\' + 'ycrcb_' + f)
                    image       = cv2.imread(datasetpath + '\\' + directories[d] + '\\' + 'ycrcb_' + f) 
                    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # convert to YCrCb
                    cv2.imwrite(datasetpath + '\\' + directories[d] + '\\' + 'ycrcb_' + f, ycrcb_image)
        else:
            onlyfiles = [f for f in listdir(datasetpath) if isfile(join(datasetpath, f))]
            for entry in onlyfiles:                    
                f = str(entry)
                #f = f.replace('<DirEntry ','')
                #f = f.replace('>','')
                f = f.replace("'",'')
                print('Augmenting File = ',  f)
                if (flip_h == True):
                    shutil.copyfile(datasetpath + '\\'  + f, datasetpath + '\\'  + 'flipped_h_' + f)
                    image = cv2.imread(datasetpath + '\\'  + 'flipped_h_' + f)
                    image = cv2.flip(image, 0)
                    cv2.imwrite(datasetpath + '\\' + 'flipped_h_' + f, image)
                if (flip_v == True):
                    shutil.copyfile(datasetpath + '\\'  + f, datasetpath + '\\'  + 'flipped_v_' + f)
                    image = cv2.imread(datasetpath + '\\'  + 'flipped_v_' + f)
                    image = cv2.flip(image, 1)
                    cv2.imwrite(datasetpath + '\\' + 'flipped_v_' + f, image)
                if (flip_hv == True):
                    shutil.copyfile(datasetpath + '\\'  + f, datasetpath + '\\'  + 'flipped_hv_' + f)
                    image = cv2.imread(datasetpath + '\\'  + 'flipped_hv_' + f)
                    image = cv2.flip(image, -1)
                    cv2.imwrite(datasetpath + '\\' + 'flipped_hv_' + f, image)
                if (gaussian_noise == True):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'g_noise_' + f)
                    image   = cv2.imread(datasetpath + '\\'  + 'g_noise_' + f)
                    g_noise = cv2.imread(datasetpath + '\\'  + 'g_noise_' + f)
                    mean    = ( 100,  100,  100) 
                    sigma   = ( 200,  200,  200)
                    cv2.randn(g_noise, mean, sigma)
                    image = g_noise + image
                    cv2.imwrite(datasetpath + '\\' + 'g_noise_' + f, image)
                if (crop == True):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'crop_' + f)  
                    image   = cv2.imread(datasetpath + '\\'  + 'crop_' + f)
                    y1 = randint(0, int(image.shape[0]/2))
                    x1 = randint(0, int(image.shape[1]/2))
                    y2 = randint(int(image.shape[0]/2), image.shape[0])
                    x2 = randint(int(image.shape[1]/2), image.shape[1])
                    image = image[y1:y2, x1:x2]
                    crop_img = image[y1:y2, x1:x2]
                    height, width, depth = image.shape
                    image = cv2.resize(crop_img, (height, width))
                    cv2.imwrite(datasetpath + '\\' + 'crop_' + f, image)
                if (translation == True):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'translated_' + f)  
                    image   = cv2.imread(datasetpath + '\\'  + 'translated_' + f)
                    x = randint(-int(image.shape[0]/3), int(image.shape[0]/3))
                    y = randint(-int(image.shape[1]/3), int(image.shape[1]/3))
                    M = np.float32([[1, 0, x],[0, 1, y]])
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) 
                    cv2.imwrite(datasetpath + '\\' + 'translated_' + f, image)
                if (sharpen == True):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'sharpen_' + f)  
                    image   = cv2.imread(datasetpath + '\\'  + 'sharpen_' + f)
                    kernel_sharpening = np.array([[-5, -5, -5],  [-5, 45, -5], [-5, -5, -5]])
                    image = cv2.filter2D(image, -1, kernel_sharpening)
                    cv2.imwrite(datasetpath + '\\' + 'sharpen_' + f, image)
                if (emboss == True):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'emboss_' + f)  
                    image   = cv2.imread(datasetpath + '\\'  + 'emboss_' + f)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                    kernel_embossing = np.array([[0, -1, -1],  [1, 0, -1], [1, 1, 0]])
                    image = cv2.filter2D(image, -1, kernel_embossing)
                    cv2.imwrite(datasetpath + '\\' + 'emboss_' + f, image)
                if (change_color == 'yuv' or change_color == 'all'):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'yuv_' + f)
                    image     = cv2.imread(datasetpath + '\\' + 'yuv_' + f) 
                    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    cv2.imwrite(datasetpath + '\\' + 'yuv_' + f, yuv_image)
                if((change_color == 'hsv' or change_color == 'all')):
                    shutil.copyfile(datasetpath + '\\' + f, datasetpath + '\\' + 'hsv_' + f)
                    image     = cv2.imread(datasetpath + '\\' + 'hsv_' + f) 
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    cv2.imwrite(datasetpath + '\\' + 'hsv_' + f, hsv_image)
                if((change_color == 'ycrcb' or change_color == 'all')):
                    shutil.copyfile(datasetpath + '\\'  + f, datasetpath + '\\' + 'ycrcb_' + f)
                    image       = cv2.imread(datasetpath  + '\\' + 'ycrcb_' + f) 
                    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # convert to YCrCb
                    cv2.imwrite(datasetpath + '\\' + 'ycrcb_' + f, ycrcb_image)
            break
    elapsed_time = time.time() - start_time
    print('Done! Total Time = ' + time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
    return

# Usage: Data x10
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', flip_h = False, flip_v = True, flip_hv = False, gaussian_noise = True, crop = True, translation = True, sharpen = True, emboss = True, change_color = 'all')

# Usage: Data x48
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', flip_v = True)
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', translation = True)
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', sharpen = True, emboss = True)
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', change_color = 'ycrcb')
#data_augmentation(datasetpath = 'D:\\SL-Recognition\\Dataset-02-ASL\\00', gaussian_noise = True)

############################################################################
    
# Function: Validation Prediction
def validation_predicition(model, path, batch_size, image_size_x = 224, image_size_y = 224, rows = 5, cols = 5):
    pred_list   = []
    title_list  = []
    image_list  = []
    datagen     = image.ImageDataGenerator(rescale = 1./255) # Generator for Prediction
    generator   = datagen.flow_from_directory(path, target_size = (image_size_x, image_size_y), batch_size = batch_size, class_mode = 'categorical', shuffle = False) 
    fnames      = generator.filenames # Get Filenames
    true_labels = generator.classes # Get True Labels
    label2index = generator.class_indices # label Index
    idx2label   = dict((v,k) for k,v in label2index.items()) # Map Class Index and Label
    prediction  = model.predict_generator(generator, steps = generator.samples/generator.batch_size, verbose = 1)
    pred_class  = np.argmax(prediction, axis = 1)
    errors      = np.where(pred_class != true_labels)[0]
    limit = len(errors)
    for i in range(0, limit):
        p_class  = np.argmax(prediction[errors[i]])
        p_label  = idx2label[p_class]
        print('True Label: {}, Prediction: {}, Confidence: {:.2%}'.format(fnames[errors[i]].split('/')[0], p_label, prediction[errors[i]][p_class]))
        pred_list.append('True Label: {}, Prediction: {}, Confidence: {:.2%}'.format(fnames[errors[i]].split('/')[0], p_label, prediction[errors[i]][p_class]))
        conf_value = float(prediction[errors[i]][p_class])*100
        conf_value = "{0:.2f}".format(conf_value)
        original   = image.load_img('{}/{}'.format(path, fnames[errors[i]]))
        title_list.append(str(p_label) + ' (' +  conf_value + ' %)' )
        image_list.append(original)              
    print('Total Number of Errors = {} of {} samples'.format(len(errors), generator.samples))
    fig = plt.figure(figsize = [10, 10])
    for i in range(0, len(pred_list)):
        if ((i + 1) > rows*cols):
            break
        fig.add_subplot(rows, cols, i + 1 )
        plt.axis('off')
        plt.title(title_list[i])
        plt.imshow(image_list[i])
    return pred_list

# Usage
# prediction_valid = validation_predicition(model = cnn, path = path_output_2, batch_size = valid_set.batch_size, image_size_x = 224, image_size_y = 224, rows = 10, cols = 5)

############################################################################

# Function: Directory Prediction
def dir_prediction(model, datasetpath, directories, image_size_x = 224, image_size_y = 224, rows = 5, cols = 5):
    pred_list   = []
    title_list  = []
    image_list  = []
    #result_list = []
    for dirpath, dirnames, filenames in os.walk(datasetpath):
        for i in range(0, len(filenames)):
            file_path  = datasetpath + '\\' + filenames[i]
            test_image = image.load_img(file_path, target_size = (image_size_x, image_size_y))
            test_image = image.img_to_array(test_image)
            test_image = test_image/255.0
            test_image = np.expand_dims(test_image, axis = 0)
            result     = model.predict_proba(test_image)
            print(directories[np.argmax(max(result))] + ' = ' + str(np.round(result[0][np.argmax(max(result))], 4)))
            pred_list.append(directories[np.argmax(max(result))] + ' = ' + str(np.round(result[0][np.argmax(max(result))], 4)))
            conf_value = float(result[0][np.argmax(max(result))])*100
            conf_value = "{0:.2f}".format(conf_value)
            title      = (directories[np.argmax(max(result))] + ' (' + conf_value + '%)') 
            original   = image.load_img(file_path)
            title_list.append(title)
            image_list.append(original)
            #result = result.T
            #result = pd.DataFrame(result, columns = ['Probability'], index = directories)
            #result_list.append(result)
    fig = plt.figure(figsize = [10, 10])
    for i in range(0, len(pred_list)):
        if ((i + 1) > rows*cols):
            break
        fig.add_subplot(rows, cols, i + 1 )
        plt.axis('off')
        plt.title(title_list[i])
        plt.imshow(image_list[i])
    return pred_list

# Usage
#class_list     = os.listdir(path_output_1)
#path_directory = 'D:\\SL-Recognition\\Dataset-02-ASL\\00'

#prediction_dir = dir_prediction(model = cnn, datasetpath = path_directory, directories = class_list, image_size_x = 224, image_size_y = 224, rows = 2, cols = 3)

############################################################################

# Function: Single Prediction
def single_prediction(model, path, directories, image_size_x = 224, image_size_y = 224):
    pred_list  = []
    test_image = image.load_img(path, target_size = (image_size_x, image_size_y))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    result     = model.predict_proba(test_image)
    for i in range(0, len(directories)):
        print(directories[i] + ' = ' + str(np.round(result[0][i], 4)))
        pred_list.append(directories[i] + ' = ' + str(np.round(result[0][i], 4)))
    conf_value = float(result[0][np.argmax(max(result))])*100
    conf_value = "{0:.2f}".format(conf_value)
    title      = ('Class Prediction = ' + directories[np.argmax(max(result))] + ' (' + conf_value + '%)')      
    original   = image.load_img(path)
    plt.figure(figsize = [10, 10])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
    return pred_list

# Usage
#class_list = os.listdir(path_output_1)
#path_image = 'D:\\SL-Recognition\\Dataset-02-ASL\\dataset\\validation\\R\\R8 (5).jpg'

#prediction_single = single_prediction(model = cnn, path = path_image, directories = class_list, image_size_x = 224, image_size_y = 224)
 
############################################################################
