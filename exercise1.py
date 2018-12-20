# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:04:51 2018

@author: 우람
"""

# Example 1: Multi-class classification
#
#  Classify stocks into deciles on their returns
#  Features: past 3, 6, 12 month returns
#  y: class label (0, ..., 9) based on the future 1 month return.

import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import pickle
import os
import tensorflow as tf
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\알고\\practice')

#############################################
# Get data
#############################################
sample_universe = pickle.load(open("data/sample1.pickle", "rb"))
sess=tf.Session()
sess.run(tf.global_variables_initializer())

x = {}
y = {}
ret1m = {}
for m in range(36):
    current_sample = sample_universe[m]
    x[m] = current_sample.loc[:, ['ret3', 'ret6', 'ret12']]
    y[m] = current_sample.loc[:, 'label']
    ret1m[m] = current_sample.loc[:, 'target_ret_1']

# Split the sample
# Training set
x_tra = np.concatenate([v for k, v in x.items() if k < 12])
y_tra = np.concatenate([v for k, v in y.items() if k < 12])
# Validation set
x_val = np.concatenate([v for k, v in x.items() if k >= 12 if k < 24])
y_val = np.concatenate([v for k, v in y.items() if k >= 12 if k < 24])
# Test set
x_tes = np.concatenate([v for k, v in x.items() if k >= 24 if k < 36])
y_tes = np.concatenate([v for k, v in y.items() if k >= 24 if k < 36])

#############################################
# Train the model
#############################################
# Model building
num_layer = 5
num_neuron = [64, 32, 16, 8, 4]
activation = 'relu'
optimizer = 'adam'
dropout_rate = 0
l1_norm = 0
num_class = 10

input = Input(shape=(x_tra.shape[1],)) #input layer 구성
hidden = input
for i in range(num_layer): #hidden layer를 하나씩 생성해서 쌓아주기 
    #num_neuron 몇개의 뉴런을 만들건지 
    #l1 regularization = Lasso.. 오버피팅을 피하기 위한 방법. 하나의 인풋에 너무 많은 웨이트를 주지 않기 위함 
    hidden = Dense(num_neuron[i], activation=activation, kernel_regularizer=regularizers.l1(l1_norm))(hidden)
    hidden = Dropout(dropout_rate)(hidden)
output = Dense(num_class, activation='softmax')(hidden) #output에선 softmax를 써야한다. 
model = Model(inputs=input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Normalize input
x_mean = x_tra.mean(axis=0)
x_std = x_tra.std(axis=0)  #x_val이든, x_tes이든 정규화할 때 평균과 표준편차는 x_tra로 한다!! 실제론 미래의 mean과 std를 모르니까'''
x_tra = (x_tra - x_mean) / x_std
x_val = (x_val - x_mean) / x_std
x_tes = (x_tes - x_mean) / x_std

# Fit the model: early stopping based on validation loss
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2, verbose=2)
model.fit(x_tra, y_tra, validation_data=(x_val, y_val), epochs=25,
               batch_size=32, callbacks=[early_stop], verbose=2) #메모리 낭비를 막기위해서 배치사이즈는 보통 2의 배수로 한다..

# Evaluate
loss_tra, acc_tra = model.evaluate(x_tra, y_tra, verbose=0)
loss_val, acc_val = model.evaluate(x_val, y_val, verbose=0)
loss_tes, acc_tes = model.evaluate(x_tes, y_tes, verbose=0)

stat_performance = pd.DataFrame({'Train':[loss_tra, acc_tra], 'Valid':[loss_val, acc_val], 'Test':[loss_tes, acc_tes]} ,index=['loss', 'acc'])
print(stat_performance)

# Test set class returns
class_return = np.zeros([12,10])
class_count = np.zeros([12,10]) #매 월 몇 개의 주식이 들어가 있는지
for m in range(24,36):
    x_m = (x[m] - x_mean) / x_std #normalize! 
    class_prob = model.predict(x_m)
    pre_class = class_prob.argmax(axis=-1)
    for cls in range(num_class):
        idx = pre_class == cls
        if any(idx):
            class_return[m-24][cls] = np.mean(ret1m[m].values[pre_class == cls])
            class_count[m-24][cls] = np.sum([pre_class == cls])

print(pd.DataFrame(class_count))
print(pd.DataFrame(class_return))
r_mean = np.mean(class_return, axis=0)
r_std = np.std(class_return, axis=0)
financial_performance = pd.DataFrame([r_mean, r_std, r_mean/r_std], index=['mean','std','sr'])
print(financial_performance)
tf.reset_default_graph()
