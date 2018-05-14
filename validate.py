#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:27:22 2018

@author: msrk
"""

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from model import create_model

from matplotlib.pyplot import imshow


def load_trained(weights):
    model = create_model()
    model.load_weights(weights)
    return model
    
        

def load_im(path):
    img = load_img(path,target_size = (150,150))
    img = img_to_array(img)
    imshow(img)
    img = np.expand_dims(img,0)
    return img

