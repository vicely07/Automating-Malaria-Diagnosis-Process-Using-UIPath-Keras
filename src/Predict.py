# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:03:48 2019

@author: Mango
"""

from keras.models import load_model
import numpy as np
import os
import cv2
def convert_to_array(img):
    img = cv2.imread(img)
    #img_ = Image.fromarray(im, 'RGB')
    image = cv2.resize(img, (50, 50))
    return np.array(image)
def get_cell_name(label):
    if label==0:
        return "Infected with Malaria"
    if label==1:
        return "Uninfected"
def predict_cell(file):
    model = load_model('C:/Users/Mango/Documents/Malaria-UIPath/src/cells.h5')
    print("Predicting Type of Cell Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    #label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    label_index=np.argmax(score)
    acc=np.max(score)
    loss=np.min(score)
    Cell=get_cell_name(label_index)
    return Cell, str("Accuracy: "+str(round(acc*100,1))+"%, Loss: "+str(round(loss*100,1))+"%")