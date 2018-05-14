import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image as PImage
import glob
import imageio

def imgDescription():
    for image_path in glob.glob("datasets/*.jpg"):
        im = imageio.imread(image_path)
        print (im.shape)
        print (im.dtype)
        

img = os.listdir('datasets')
img = ['datasets/'+i for i in img]
df = pd.DataFrame(img, columns = ['x'])

y = [int(str.rfind(i,'mango')!=-1) for i in img]

df['y'] = y
print(df)

df.to_csv("data.csv")


