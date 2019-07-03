
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import MobileNetV2
from keras.applications import NASNetMobile 
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.client import device_lib 

print(device_lib.list_local_devices())
# In[2]:


#base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
#base_model=MobileNetV2(weights='imagenet',include_top=False)
base_model=NASNetMobile(weights='imagenet',include_top=False) #imports the NASNetMobile model and discards the last 1000 neuron layer.


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation


# In[3]:


model=Model(inputs=base_model.input,outputs=preds)
#Training models with weights merge on GPU (recommended for NV-link)
#print(tf.list_devices())

try:
    parallel_model = multi_gpu_model(model, gpus=2, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..") 

#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:


#for layer in model.layers[:20]:
#    layer.trainable=False
#for layer in model.layers[20:]:
#    layer.trainable=True


# In[5]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


# In[33]:


#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
parallel_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.summary()

MODEL_FILE = 'model.h5'
step_size_train=train_generator.n//train_generator.batch_size
#model.fit_generator(generator=train_generator,
#                   steps_per_epoch=step_size_train,
#                   epochs=5)
parallel_model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)

model.save(MODEL_FILE)

# Prediction of the custom model
#model = load_model(MODEL_FILE)
img = image.load_img('train/dogs/images.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print (preds[0])
predicted_class_indices=np.argmax(preds,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)

img = image.load_img('train/horses/images.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
predicted_class_indices=np.argmax(preds,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)

