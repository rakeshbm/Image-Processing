# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:53:46 2019

@author: nisha
"""

import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dense
from keras import optimizers



def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
    if type(ims[0]) is np.ndarray:
        ims=np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims=ims.transpose((0,2,3,1))
    f=plt.figure(figsize=figsize)
    cols=len(ims)//rows if len(ims)%2==0 else len(ims)//rows+1
    for i in range(len(ims)):
        sp=f.add_subplot(rows,cols,i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else 'none')
        
        
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix: Without Normalization")
    print(cm)
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

        
        
def get_Model():        
    train_path='mixup/train'
    test_path='random_erasing/test'
    valid_path='random_erasing/valid'
    
    
    train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['door','traffic','building','window'],batch_size=10)
    valid_batches=ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['door','traffic','building','window'],batch_size=4)
    test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['door','traffic','building','window'],batch_size=10)
    
    imgs,labels=next(train_batches)
    plots(imgs,titles=labels)
    
    vgg16_model = VGG16(weights='imagenet', include_top=True)
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
    for layer in model.layers:
        layer.trainable=False
    
    
    model.add(Dense(4, activation='softmax'))
    model.compile(keras.optimizers.Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches,steps_per_epoch=40, validation_data=valid_batches, validation_steps=40,epochs=65,verbose=2)
    
    cm_plot_labels=['door','traffic','building','window']
    
    
    #plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
    model.fit_generator(valid_batches,steps_per_epoch=40, validation_data=train_batches, validation_steps=40,epochs=65,verbose=2)
    
    
    from keras.preprocessing.image import load_img
    # load an image from file
    image = load_img('s.jpg', target_size=(224, 224))
        
    from keras.preprocessing.image import img_to_array
    # convert the image pixels to a numpy array
    
    image = img_to_array(image)
    
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    from keras.applications.vgg16 import preprocess_input
    
    # prepare the image for the VGG model
    image = preprocess_input(image)
     
    # predict the probability across all output classes
    yhat = model.predict(image)
    
    maxarg=np.argmax(yhat)
    
    print(cm_plot_labels[maxarg]," : ",max(max(yhat)*100))
    fname = "vgg-weight_mixup2.h5"
    
    model.save_weights(fname,overwrite=True)
    
    return model



get_Model()
#tensorflowproject -- environment