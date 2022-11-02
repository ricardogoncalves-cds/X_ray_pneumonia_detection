# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:05:39 2022

@author: Ricardo
"""

"""
The dataset used in this project can be found at https://www.kaggle.com/tolgadincer/labeled-chest-xray-images

Code used to resize the original images

import cv2
import glob

i=0
image_list=[]
resized_list=[]

for filename in glob.glob(r"C:\chest_xray\test\NORMAL\*.jpeg"):
    img=cv2.imread(filename)
    image_list.append(img)
    
for image in image_list:
    resized=cv2.resize(image, (512,512))
    resized_list.append(resized)
    
    cv2.imwrite(r"C:\chest_xray\test\NORMAL_RESIZED\*.jpeg" %i, (resized))
    i+=1   
"""

import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
import tensorflow as tf
import random
import glob
import cv2

from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
NO_EPOCH = 50 
SEED = 55
seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

"""
# NOTE that the files are being read from Google Drive.
# Set your own directory before running this code

train_normal=glob.glob(r"/content/gdrive/MyDrive/dataset/train/normal_resized/*.jpeg")
train_pneumonia=glob.glob(r"/content/gdrive/MyDrive/dataset/train/pneumonia_resized/*.jpeg")

test_normal=glob.glob(r"/content/gdrive/MyDrive/dataset/test/normal_resized/*.jpeg")
test_pneumonia=glob.glob(r"/content/gdrive/MyDrive/dataset/test/pneumonia_resized/*.jpeg")
"""

# Creating train dataframe

train_images=[img for img in train_normal]
train_images.extend([img for img in train_pneumonia])

data_train=pd.DataFrame(np.concatenate([['Normal']*len(train_normal) , ['Pneumonia']*len(train_pneumonia)]), columns = ['class'])
data_train['image'] = [img for img in train_images]

# Creating test dataframe

test_images=[img for img in test_normal]
test_images.extend([img for img in test_pneumonia])

data_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])
data_test['image'] = [img for img in test_images]

"""
### DATA EXPLORATION

### Train Barplot

plt.figure(figsize=(6,2))

ax = sns.countplot(x='class', data=data_train)

plt.title("Training set distribution", fontsize=16)
plt.xlabel("Class", fontsize= 11)
plt.ylabel("Samples", fontsize= 11)
plt.ylim(0,5000)
plt.xticks([0,1], ['Normal', 'Pneumonia'], fontsize = 8)

for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.35, p.get_height()+150), fontsize=8)
    
plt.show()

### Train Pie-chart

plt.figure(figsize=(5,5))

data_train['class'].value_counts().plot(kind='pie',labels = ['',''], autopct='%1.1f%%', textprops = {"fontsize":12})

plt.title('Train set distribution', fontsize=16)
plt.show()

### Test Barplot

plt.figure(figsize=(6,2))

ax = sns.countplot(x='class', data=data_test)

plt.title("Testing set distribution", fontsize=16)
plt.xlabel("Class", fontsize= 11)
plt.ylabel("Samples", fontsize= 11)
plt.ylim(0,500)
plt.xticks([0,1], ['Normal', 'Pneumonia'], fontsize = 8)

for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.35, p.get_height()+20), fontsize = 8)
    
plt.show()

### Test Pie-chart

plt.figure(figsize=(5,5))

data_test['class'].value_counts().plot(kind='pie',labels = ['',''], autopct='%1.1f%%', textprops = {"fontsize":12})

plt.title('Test set distribution', fontsize=16)
plt.legend(labels=['Pneumonia', 'Normal'])
plt.show()

### Healthy Cases Sample

print('Healthy cases sample')

plt.figure(figsize=(16,16))

for i in range(0, 3):
    plt.subplot(1,3,i + 1)
    img = cv2.imread(train_normal[i])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()

plt.show()

### Pneumonia Cases Sample

print('Pneumonia cases sample')

plt.figure(figsize=(16,16))

for i in range(0, 3):
    plt.subplot(1,3,i + 1)
    img = cv2.imread(train_pneumonia[i])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()

plt.show()
"""

# Data Preparation

train_df, val_df = train_test_split(data_train, test_size = 0.20, random_state = SEED, stratify = data_train['class'])

train_datagen=ImageDataGenerator(rescale=1/255.)
val_datagen=ImageDataGenerator(rescale=1/255.)

train_ds=train_datagen.flow_from_dataframe(train_df, x_col='image', y_col='class', target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE, seed=SEED)
validation_ds=val_datagen.flow_from_dataframe(val_df, x_col='image', y_col='class', target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE, seed = SEED)
test_ds=val_datagen.flow_from_dataframe(data_test, x_col='image', y_col='class', target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=1, shuffle=False)

# Callbacks

early_stop=callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0000001, restore_best_weights=True)
plateau=callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_delt=0.0000001, cooldown=0, verbose=1) 

# Model

def make_model():
    
    model = keras.models.Sequential()
    model.add(Conv2D(16, (3,3), padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), padding='valid'))
    model.add(Conv2D(64, (3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

# Model Compile // Note that we are using Adam as the Optimizer in this intance but it can be changed to SGD if desired

model = make_model()
model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.00003), metrics='binary_accuracy')

# Model fitting

history = model.fit(train_ds, batch_size = BATCH_SIZE, epochs = NO_EPOCH, validation_data=validation_ds, callbacks=[early_stop, plateau], steps_per_epoch=(len(train_df)/BATCH_SIZE), validation_steps=(len(val_df)/BATCH_SIZE));

# Scores

score_val = model.evaluate(validation_ds, steps = len(val_df)/BATCH_SIZE, verbose = 0)
score_test = model.evaluate(test_ds, steps = len(test_ds), verbose = 0)
print('Validation loss:', score_val[0])
print('Validation accuracy:', score_val[1])
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])

"""
### Learning Curve Plots

### Loss Plot

fig, ax = plt.subplots(figsize=(20,8))
sns.lineplot(x = history2.epoch, y = history2.history['loss'])
sns.lineplot(x = history2.epoch, y = history2.history['val_loss'])
ax.set_title('Learning Curve (Loss)')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.set_ylim(0, 0.5)
ax.legend(['train', 'val'], loc='best')
plt.show()

### Accuracy Plot

fig, ax = plt.subplots(figsize=(20,8))
sns.lineplot(x = history2.epoch, y = history2.history['binary_accuracy'])
sns.lineplot(x = history2.epoch, y = history2.history['val_binary_accuracy'])
ax.set_title('Learning Curve (Accuracy)')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylim(0.8, 1.0)
ax.legend(['train', 'val'], loc='best')
plt.show()
"""
