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
import copy
import gridfs
import geocoder

import pymongo
   

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
            
def call_sliding_window(FLAGS):

    try:
        image_file = FLAGS.image
        model_weights = FLAGS.model
    except:
        print("Could not parse parser arguments. Please try again.")
    
    # load the image and define the window width and height
    image = cv2.imread((image_file))
    input_image = image.deepcopy()
    
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
    #json_file = open('model.json', 'r')
    '''
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    '''
    model.load_weights(model_weights)
    
    maxy=90
    i=0
    bb_coord = []
    # loop over the image pyramid
    #for resized in pyramid(image, scale=1.5):
    while i==0:
        #i=1
        detected_objects = []

        clone = image.copy()
        maxy=85
    	# loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
    		# if the window does not meet our desired window size, ignore it
    		#if window.shape[0] != winH or window.shape[1] != winW:
    			#continue
     
    		
            
            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            image1= image[y:y+winH, x:x + winW]
            #image1=img_to_array(image1)
            image1= cv2.resize(image1,(224,224),3)
            #print(image1.shape)
            #image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
            #np.reshape(image1, (224,224,3))
            image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
            #print(image1.shape)
            image1 = preprocess_input(image1)
            #yhat = model.predict(image)
            #if(max(yhat)>=0.5):
                #cv2.imshow("Window", clone)
            #predict the probability across all output classes
            yhat = model.predict(image1)
            #yhat=model.predict(clone)
            #maxarg=np.argmax(yhat)
            
            if max(max(yhat)*100) > maxy:
                
                detected_object = clone[y:y+winW, x:x+winH].deepcopy()
                detected_objects.append(detected_object)
                
                temp = [(x,y),(x+winW, y+winH)]
                bb_coord.append(temp)
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        		# since we do not have a classifier, we'll just draw the window
                maxy=max(max(yhat)*100)
                #print(max(max(yhat)*100))
                #cv2.imshow("i"+str(i), clone)
                #i+=1
        
        # processed_image_information, mydict = {'_id': <current location>, 'location': <current location>, 'input_image': <given input image ndarray>, 'bounding_box_coordinates': <bounding box coordinates of detected objects>, 'detected objects': <list of ndarrays of detected objects>}
        database_dict = { "_id": geocoder.ip('me').latlng, "location": geocoder.ip('me').latlng, "image": input_image, "bounding_box_coordinates": bb_coord, "detected_objects": detected_objects }
            
        i=i+1
        return database_dict
#       time.sleep(0.025)
