#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:07:06 2018

@author: msrk
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 226, 200, 3

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('datasets/mango.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
x = x.reshape((1,) + x.shape+(1,))  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview/', save_prefix='mango', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely