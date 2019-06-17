# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:15:21 2019

@author: nisha
"""
# import the necessary packages
import imutils
import argparse
import keras
import time
import cv2
import numpy as np
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense,Flatten
import json
 
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image
        
        
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and define the window width and height
image = cv2.imread(args["image"])

(winW, winH) = (128, 128)
from keras.models import model_from_json
vgg16_model = VGG16(weights='imagenet', include_top=True)
vgg16_model.layers.pop()
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
for layer in model.layers:
    layer.trainable=False    
model.add(Dense(4, activation='softmax'))
model.compile(keras.optimizers.Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("vgg-weight_mixup2.h5")

maxy=90
i=0

cm_plot_labels=['door','traffic','building','window']

data={}
data['annotation']={}
data['annotation']['folder']='dr'
data['annotation']['filename']=args["image"]
data['annotation']['object']=[]

data['annotation']['size']={}
data['annotation']['size']['width']=image.shape[0]
data['annotation']['size']['depth']=3
data['annotation']['size']['height']=image.shape[1]

s='../dr/'+args["image"]
data['annotation']['path']=str(s)

j=0

while i==0:
    
    clone = image.copy()
    maxy=80
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		#if window.shape[0] != winH or window.shape[1] != winW:
			#continue
 
		
        image1= image[y:y+winH, x:x + winW]
        
        image1= cv2.resize(image1,(224,224),3)
        
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        
        image1 = preprocess_input(image1)
        
        #predict the probability across all output classes
        yhat = model.predict(image1)
        
        
        if max(max(yhat)*100) > maxy:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    		# since we do not have a classifier, we'll just draw the window
            maxy=max(max(yhat)*100)
            p={}
            p['bndbox']={}
            p['bndbox']['xmax']=x+winW
            p['bndbox']['ymin']=y
            p['bndbox']['ymax']=y+winH
            p['bndbox']['xmin']=x
            p['name']=cm_plot_labels[np.argmax(yhat)]
            data['annotation']['object'].append(p)
            
            #crop detected objects
            cv2.imwrite('img'+str(j)+'.jpg',image[y:y+winH, x:x + winW])
            j+=1

    cv2.imshow("Window", clone)
    cv2.imwrite('window'+str(i)+'.jpg',clone)
    i=i+1
    cv2.waitKey(1)
    time.sleep(0.025)
    with open(args["image"].replace('.','_')+".json", 'w') as outfile:  
        json.dump(data, outfile)