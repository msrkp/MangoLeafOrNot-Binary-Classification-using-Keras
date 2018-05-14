#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:20:12 2018

@author: msrk
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def generateData():
    batch_size = 1
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'preview/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'preview/test',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    return train_generator, validation_generator,batch_size







