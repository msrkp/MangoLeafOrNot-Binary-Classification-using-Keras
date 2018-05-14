#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:01:50 2018

@author: msrk
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from batchGenerator import generateData


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    
    model.add(Conv2D(32, (3, 3),dim_ordering="th"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model
    
def train():
    model = create_model()    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    train_generator, validation_generator, batch_size = generateData()
    model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=500 // batch_size)
   
    model.save_weights('first_try.h5')  # always save your weights after training or during training
    
    return model
