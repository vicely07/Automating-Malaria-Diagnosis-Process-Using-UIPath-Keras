# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 01:15:03 2019

@author: Mango
"""
import sys 
import os
import cv2
import numpy as np
import Predict



Predict = Predict
if __name__=="__main__":
    value = (input("enter input here :")).replace("'", "")
    output = str(Predict.predict_cell(value)).replace("(", "").replace(")", "").replace("'", "")
    logout = str(output).replace("Accuracy:", "").replace("Loss:", "")
    print(logout)
    log = open('C:/Users/Mango/Documents/Malaria-UIPath/Result/log.csv', 'a')
    log.write("\n" + str(logout))
    log.close()
    result = open('C:/Users/Mango/Documents/Malaria-UIPath/Result/lastresult.txt',"w") 
    print(result)
    result.write(str(output))
    result.close()
    print(output)