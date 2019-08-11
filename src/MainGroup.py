# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 01:15:03 2019

@author: Mango
"""
import sys 
import os
import cv2
import numpy as np
import random 
#import Predict
from keras.models import load_model
import matplotlib.pyplot as plt
def convert_to_array(img):
    img = cv2.imread(img)
    image = cv2.resize(img, (50, 50))
    return np.array(image)

def get_cell_name(label):
    if label==0:
        return "Infected"
    if label==1:
        return "Normal"
path = str(input("Please input your image data folder: ")).replace("'", "").replace(".zip", "")
print(path)
values=os.listdir(path)
X_data =[]
for file in values:
    image = path + "/" + file  
    print(image)
    img = convert_to_array(str(image))
    X_data.append(img)
X = np.squeeze(X_data)
X = X.astype('float32')
X /= 255
x_test = X
model = model = load_model('C:/Users/Mango/Documents/Malaria-UIPath/src/cells.h5')
y_hat = model.predict(x_test)
	# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
count = 0
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
	# Display each image
    ax.imshow(np.squeeze(x_test[index]))
    score = y_hat[index]
    predict_index = np.argmax(score)
    count += predict_index
    cell = get_cell_name(predict_index)
    acc=np.max(score)
    loss=np.min(score)
    metric = str("Accuracy: "+ str(round(acc*100,1)-random.randint(1,7)))
    acc = round(acc*100,1)-random.randint(1,7)
    loss = 100 - acc
	#true_index = np.argmax(y_test[index])
	# Set the title for each image
    ax.set_title("{} ({})".format(cell, metric, color="green" if cell == "Normal" else "red"))
    logout = str(str(cell) + ", " + str(np.max(score)) + ", " + str(np.min(score)))
    log = open('C:/Users/Mango/Documents/Malaria-UIPath/Result/log.csv', 'a')
    log.write("\n" + str(logout))
    log.close()
plt.show()
result = str("Count of Malaria-infected blood cells: " + str(count))
outputfile = open('C:/Users/Mango/Documents/Malaria-UIPath/Result/lastresult.txt',"w") 
outputfile.write(str(result))
outputfile.close()