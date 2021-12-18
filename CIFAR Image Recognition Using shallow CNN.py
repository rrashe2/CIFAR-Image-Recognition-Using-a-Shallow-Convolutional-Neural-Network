# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 02:28:02 2021

@author: rahiy
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



print('x_train:\t{}' .format(x_train.shape))
print('y_train:\t{}' .format(y_train.shape))
print('x_test:\t\t{}'.format(x_test.shape))
print('y_test:\t\t{}'.format(y_test.shape))



for i in range(9):
	# define subplot
	plt.subplot(3, 3, i+1)
	# plot raw pixel data
	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show() 


x_train_ds = tf.data.Dataset.from_tensor_slices(x_train)
y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
train_ds = tf.data.Dataset.zip((x_train_ds, y_train_ds))

x_test_ds = tf.data.Dataset.from_tensor_slices(x_test)
y_test_ds = tf.data.Dataset.from_tensor_slices(y_test)
test_ds = tf.data.Dataset.zip((x_test_ds, y_test_ds))


BATCH_SIZE = 256

def preprocess_fn(image, label):
 image = tf.cast(image, dtype=tf.float32)
 image = tf.image.per_image_standardization(image)
 return(image, label)

train_ds = train_ds.map(preprocess_fn)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).repeat()

def preprocess_fn(image, label):
 image = tf.cast(image, dtype=tf.float32)
 image = tf.image.per_image_standardization(image)
 return(image, label)

test_ds = test_ds.map(preprocess_fn)
test_ds = test_ds.shuffle(1000).batch(BATCH_SIZE)

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = [32,32,3]))
model.add(MaxPooling2D(pool_size=2,strides=2, padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2,strides=2, padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units= 256,activation='relu'))
model.add(Dense(10))
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn, metrics=['sparse_categorical_accuracy'])
BATCH_SIZE = 256
model.fit(train_ds, validation_data=test_ds, steps_per_epoch=50000//BATCH_SIZE, epochs=5, validation_steps=10000//BATCH_SIZE)
loss, acc = model.evaluate(test_ds, verbose=0)
print("Accuracy", acc)

def preprocess_new_fn(image, label):
 image = tf.cast(image, dtype=tf.float32)
 image = tf.image.per_image_standardization(image)
 return(image, label)


datagen = ImageDataGenerator(horizontal_flip = True)
datagen.fit(x_train)
for i in range(9):
	# define subplot
	plt.subplot(3, 3, i+1)
	# plot raw pixel data
	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
 

x_train_ds = tf.data.Dataset.from_tensor_slices(x_train)
y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
train_ds = tf.data.Dataset.zip((x_train_ds, y_train_ds))

train_ds = tf.data.Dataset.zip((x_train_ds, y_train_ds))
train_ds = train_ds.map(preprocess_new_fn)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).repeat()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn, metrics=['sparse_categorical_accuracy'])
BATCH_SIZE = 256
model.fit(train_ds, validation_data=test_ds, steps_per_epoch=50000//BATCH_SIZE, epochs=5, validation_steps=10000//BATCH_SIZE)
loss, acc = model.evaluate(test_ds, verbose=0)
print("Accuracy", acc)

x_test = x_test[:512,:,:]
ypred = model.predict(x_test)
pred = np.argmax(ypred, axis = 1)
y_test_org = np.argmax(y_test, axis = 1)
print(confusion_matrix(y_test_org, pred ))
