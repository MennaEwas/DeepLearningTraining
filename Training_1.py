"""
Research Internship First Training 
"""
import theano
from theano import tensor
import tensorflow as tf 

''' 
Theano
Keras is a wrapper library that hides theano completely, theano is
 efficient for numerical implementation
'''
#declearing two sambolic floating-point 
a = tensor.dscalar()
b = tensor.dscalar() 

#create a simple sambolic expression 
c = a + b 
#convert it into a callable object 
f = theano.function([a,b], c)
print(f(1.5, 2.5))  #4.0

'''
introduction to tensorflow 
Nodes: data that moves bt nodes is called tensors which are multi-dimental
 arrays of real values
Edges: used for schynco 
operation: an operation is a named abstract computation which can take 
 input attributes and gave output attributes 
'''
#declearing two sambolic floating-point 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#create a simple sambolic expression 
add = tf.add(a, b)
#bind and evaluate 
sess = tf.Session()
binding = {a: 1.5, b:2.5}
c = sess.run(add, feed_dict= binding)
print(c) #4.0 
'''
Introduction to keras 
Keras: In a minimalist python library for deeplearning that can run on the top
of theano and tensorflow. It was developing to make models as fast and easy
for research and development. 

Steps for Keras 
1. Define your model --> Sequential  
2. compile your model --> compile()
3. fit your model --> fit()
4. Make prediction --> evaluate() and predict()
'''
# I was trying something

import pickle
from pathlib import Path
from skimage import io

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import csv

#2. Load images lables file  
labels = pd.read_csv("TrainLabel.csv")

#3. Make a csv file 
with open('train_arr.csv', 'w', newline='') as file: 
    writer = csv.writer(file)
    writer.writerow([i for i in range(1, 29)])

#4. open and read the file
train = pd.read_csv('train_arr.csv') 

#5. Examine RGB values in an image matrix¶
#load an image and explore -- example image is a 32x32 matrix 
example_image = io.imread('Train_Images/id_{}_label_{}.png'.format(1, labels.index[1]))
for iD in range (0, 13440):
    x = labels.iat[iD, 0]
    x = labels.index[0]
    print(iD, x)
    example_image = io.imread('Train_Images/id_{}_label_{}.png'.format(iD, x))
    ar = example_image.flatten()
    train[x] = ar
        
    


#show image
plt.imshow(example_image)
print('first' , example_image)
ar = example_image.flatten()
print('second', ar)


 



