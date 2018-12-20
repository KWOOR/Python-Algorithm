# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:22:55 2018

@author: 우람
"""

import numpy as np
import pandas as pd
import tensorflow as tf
#import get_sample as gs
from keras.models import Model
import pickle
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\알고')


def get_sample():
    train = pickle.load(open("Assignment_algo/train.pickle", "rb"))
    test = pickle.load(open("Assignment_algo/test.pickle", "rb"))

    x_train = {}
    y_train = {}
    r_train = {}
    for m in range(36):
        x_train[m] = train[m].loc[:, ['ret2', 'ret3', 'ret6', 'ret9', 'ret12']]
        y_train[m] = train[m].loc[:, 'label']
        r_train[m] = train[m].loc[:, 'target_ret_1']

    x_test = {}
    y_test = {}
    r_test = {}
    for m in range(36):
        x_test[m] = test[m].loc[:, ['ret2', 'ret3', 'ret6', 'ret9', 'ret12']]
        y_test[m] = test[m].loc[:, 'label']
        r_test[m] = test[m].loc[:, 'target_ret_1']

    return x_train, y_train, r_train, x_test, y_test, r_test

tf.reset_default_graph()
tf.get_default_session()

# Get data from get_sample.py
_x_train, _y_train, r_train, _x_test, _y_test, r_test = get_sample()

# Split the sample
# Training set (36개월 3년치 데이터)
x_train = np.concatenate([v for k, v in _x_train.items() if k < 24])
y_train = np.concatenate([v for k, v in _y_train.items() if k < 24])
# Validation set
x_valid = np.concatenate([v for k, v in _x_train.items() if k >= 24 if k < 36])
y_valid = np.concatenate([v for k, v in _y_train.items() if k >= 24 if k < 36])
# Test set
x_test = np.concatenate([v for k, v in _x_test.items()])
y_test = np.concatenate([v for k, v in _y_test.items()])

y_train = np.array([y_train]).T
y_valid = np.array([y_valid]).T
y_test = np.array([y_test]).T

# TODO 차후 MinMaxsclaer() 확인
# Normalize inputs
#x_mean = x_train.mean(axis=0)
#x_std = x_train.std(axis=0)
#x_train = (x_train - x_mean) / x_std
#x_valid = (x_valid - x_mean) / x_std
#x_test = (x_test - x_mean) / x_std

# Build the model
# TODO 각종 하이퍼 파라미터 차후 변경
num_neuron = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
num_class = 10
learning_rate = 0.001
training_epochs = 20
batch_size = 1000
keep_prob = tf.placeholder(tf.float32)

# Input place holders
X = tf.placeholder(tf.float32, [None, 47])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, num_class)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_class])

# Hidden Layer 1
#W1 = tf.get_variable("W1", shape=[5, num_neuron[0]],
#                     initializer=tf.contrib.layers.xavier_initializer())
W1 = tf.Variable(tf.random_normal([47, num_neuron[0]], stddev=0.01))
b1 = tf.Variable(tf.random_normal([num_neuron[0]]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Hidden Layer 2
#W2 = tf.get_variable("W2", shape=[num_neuron[0], num_neuron[1]],
#                     initializer=tf.keras.initializers.he_normal(seed=None))
W2 = tf.Variable(tf.random_normal([num_neuron[0], num_neuron[1]], stddev=0.01))
b2 = tf.Variable(tf.random_normal([num_neuron[1]]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Hidden Layer 3
#W3 = tf.get_variable("W3", shape=[num_neuron[1], num_neuron[2]],
#                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.Variable(tf.random_normal([num_neuron[1], num_neuron[2]], stddev=0.01))

b3 = tf.Variable(tf.random_normal([num_neuron[2]]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# Hidden Layer 4
#W4 = tf.get_variable("W4", shape=[num_neuron[2], num_neuron[3]],
#                     initializer=tf.keras.initializers.he_normal(seed=None))
W4 = tf.Variable(tf.random_normal([num_neuron[2], num_neuron[3]], stddev=0.01))
b4 = tf.Variable(tf.random_normal([num_neuron[3]]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# Hidden Layer 5
#W5 = tf.get_variable("W5", shape=[num_neuron[3], num_neuron[4]],
#                     initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.Variable(tf.random_normal([num_neuron[3], num_neuron[4]], stddev=0.01))
b5 = tf.Variable(tf.random_normal([num_neuron[4]]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.Variable(tf.random_normal([num_neuron[4], num_neuron[5]], stddev=0.01))
b6 = tf.Variable(tf.random_normal([num_neuron[5]]))
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)


W7 = tf.Variable(tf.random_normal([num_neuron[5], num_neuron[6]], stddev=0.01))
b7 = tf.Variable(tf.random_normal([num_neuron[6]]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)


W8 = tf.Variable(tf.random_normal([num_neuron[6], num_neuron[7]], stddev=0.01))
b8 = tf.Variable(tf.random_normal([num_neuron[7]]))
L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
L8 = tf.nn.dropout(L8, keep_prob=keep_prob)

# Output Layer (Hypothesis)
#W_out = tf.get_variable("output", shape=[num_neuron[4], num_class],
#                        initializer=tf.keras.initializers.he_normal(seed=None))
W_out = tf.Variable(tf.random_normal([num_neuron[7], num_class], stddev=0.01))
b_out = tf.Variable(tf.random_normal([num_class]))
logits = tf.matmul(L8, W_out) + b_out
hypothesis = tf.nn.softmax(logits)

# Loss function & Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# hypothesis = tf.nn.softmax(hypothesis)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%

buff10=x_train.copy()
buff10=pd.DataFrame(buff10).rank(axis=0)
for i in range(len(buff10.T)):
    buff10[i]=(((buff10[i]-len(buff10)*0.9)>0)*1)
buff20=x_train.copy()
buff20=pd.DataFrame(buff20).rank(axis=0)
for i in range(len(buff20.T)):
    buff20[i]=(((buff20[i]-len(buff20)*0.8)>0)*1) 
buff30=x_train.copy()
buff30=pd.DataFrame(buff30).rank(axis=0)
for i in range(len(buff30.T)):
    buff30[i]=(((buff30[i]-len(buff30)*0.7)>0)*1)
buff40=x_train.copy()
buff40=pd.DataFrame(buff40).rank(axis=0)
for i in range(len(buff40.T)):
    buff40[i]=(((buff40[i]-len(buff40)*0.6)>0)*1) 
buff50=x_train.copy()
buff50=pd.DataFrame(buff50).rank(axis=0)
for i in range(len(buff50.T)):
    buff50[i]=(((buff50[i]-len(buff50)*0.5)>0)*1)
buff60=x_train.copy()
buff60=pd.DataFrame(buff60).rank(axis=0)
for i in range(len(buff60.T)):
    buff60[i]=(((buff60[i]-len(buff60)*0.4)>0)*1) 
buff70=x_train.copy()
buff70=pd.DataFrame(buff70).rank(axis=0)
for i in range(len(buff70.T)):
    buff70[i]=(((buff70[i]-len(buff70)*0.3)>0)*1) 
buff80=x_train.copy()
buff80=pd.DataFrame(buff80).rank(axis=0)
for i in range(len(buff80.T)):
    buff80[i]=(((buff80[i]-len(buff80)*0.2)>0)*1)  
buff90=x_train.copy()
buff90=pd.DataFrame(buff90).rank(axis=0)
for i in range(len(buff90.T)):
    buff90[i]=(((buff90[i]-len(buff90)*0.1)>0)*1)  
    
empty=x_train.copy()
empty=pd.DataFrame(empty.std(axis=1))
empty1=x_train.copy()
empty1=pd.DataFrame(empty1.mean(axis=1))
x_train=np.array(pd.concat([buff10,buff20,buff30,buff40,buff50,buff60,buff70,buff80,buff90, empty, empty1], axis=1))


#x_train=np.array(pd.DataFrame(x_train).rank(axis=0)*100/len(x_train))
#x_train=np.array(pd.concat([pd.DataFrame(x_train), ((pd.DataFrame(x_train/100+1).prod(axis=1))**(1/5)-1)*100, empty, empty1], axis=1))
#x_train=x_train[:,5:]

buff10=x_valid.copy()
buff10=pd.DataFrame(buff10).rank(axis=0)
for i in range(len(buff10.T)):
    buff10[i]=(((buff10[i]-len(buff10)*0.9)>0)*1)
buff20=x_valid.copy()
buff20=pd.DataFrame(buff20).rank(axis=0)
for i in range(len(buff20.T)):
    buff20[i]=(((buff20[i]-len(buff20)*0.8)>0)*1) 
buff30=x_valid.copy()
buff30=pd.DataFrame(buff30).rank(axis=0)
for i in range(len(buff30.T)):
    buff30[i]=(((buff30[i]-len(buff30)*0.7)>0)*1) 
buff40=x_valid.copy()
buff40=pd.DataFrame(buff40).rank(axis=0)
for i in range(len(buff40.T)):
    buff40[i]=(((buff40[i]-len(buff40)*0.6)>0)*1)
buff50=x_valid.copy()
buff50=pd.DataFrame(buff50).rank(axis=0)
for i in range(len(buff50.T)):
    buff50[i]=(((buff50[i]-len(buff50)*0.5)>0)*1)
buff60=x_valid.copy()
buff60=pd.DataFrame(buff60).rank(axis=0)
for i in range(len(buff60.T)):
    buff60[i]=(((buff60[i]-len(buff60)*0.4)>0)*1)
buff70=x_valid.copy()
buff70=pd.DataFrame(buff70).rank(axis=0)
for i in range(len(buff70.T)):
    buff70[i]=(((buff70[i]-len(buff70)*0.3)>0)*1) 
buff80=x_valid.copy()
buff80=pd.DataFrame(buff80).rank(axis=0)
for i in range(len(buff80.T)):
    buff80[i]=(((buff80[i]-len(buff80)*0.2)>0)*1) 
buff90=x_valid.copy()
buff90=pd.DataFrame(buff90).rank(axis=0)
for i in range(len(buff90.T)):
    buff90[i]=(((buff90[i]-len(buff90)*0.1)>0)*1) 
    
empty=x_valid.copy()
empty=pd.DataFrame(empty.std(axis=1))
empty1=x_valid.copy()
empty1=pd.DataFrame(empty1.mean(axis=1))
x_valid=np.array(pd.concat([buff10,buff20,buff30,buff40,buff50,buff60,buff70,buff80,buff90, empty, empty1], axis=1))
#x_valid=np.array(pd.DataFrame(x_valid).rank(axis=0)*100/len(x_valid))

#buff=x_valid.copy()
#for i in range(len(buff)):
#    buff[i]=(((buff[i]-buff.mean(axis=0))>0)*1)
#buff=pd.DataFrame(buff).rank(axis=0)
#x_valid=np.array(pd.concat([pd.DataFrame(x_valid), ((pd.DataFrame(x_valid/100+1).prod(axis=1))**(1/5)-1)*100, empty, empty1], axis=1))
#x_valid=x_valid[:,5:]
#x_valid=np.array(buff)

#
#  
#empty=x_test.copy()
#empty=pd.DataFrame(empty.std(axis=1))
#empty1=x_test.copy()
#empty1=pd.DataFrame(empty1.mean(axis=1))
##buff=x_test.copy()
#for i in range(len(buff)):
#    buff[i]=(((buff[i]-buff.mean(axis=0))>0)*1)
#buff=pd.DataFrame(buff).rank(axis=0)
#x_test=np.array(pd.concat([pd.DataFrame(x_test), ((pd.DataFrame(x_test/100+1).prod(axis=1))**(1/5)-1)*100, empty, empty1], axis=1))
#x_test=x_test[:,5:]
#x_test=np.array(buff)
#
buff10=x_test.copy()
buff10=pd.DataFrame(buff10).rank(axis=0)
for i in range(len(buff10.T)):
    buff10[i]=(((buff10[i]-len(buff10)*0.9)>0)*1) 
buff20=x_test.copy()
buff20=pd.DataFrame(buff20).rank(axis=0)
for i in range(len(buff20.T)):
    buff20[i]=(((buff20[i]-len(buff20)*0.8)>0)*1) 
buff30=x_test.copy()
buff30=pd.DataFrame(buff30).rank(axis=0)
for i in range(len(buff30.T)):
    buff30[i]=(((buff30[i]-len(buff30)*0.7)>0)*1) 
buff40=x_test.copy()
buff40=pd.DataFrame(buff40).rank(axis=0)
for i in range(len(buff40.T)):
    buff40[i]=(((buff40[i]-len(buff40)*0.6)>0)*1) 
buff50=x_test.copy()
buff50=pd.DataFrame(buff50).rank(axis=0)
for i in range(len(buff50.T)):
    buff50[i]=(((buff50[i]-len(buff50)*0.5)>0)*1) 
buff60=x_test.copy()
buff60=pd.DataFrame(buff60).rank(axis=0)
for i in range(len(buff60.T)):
    buff60[i]=(((buff60[i]-len(buff60)*0.4)>0)*1) 
buff70=x_test.copy()
buff70=pd.DataFrame(buff70).rank(axis=0)
for i in range(len(buff70.T)):
    buff70[i]=(((buff70[i]-len(buff70)*0.3)>0)*1) 
buff80=x_test.copy()
buff80=pd.DataFrame(buff80).rank(axis=0)
for i in range(len(buff80.T)):
    buff80[i]=(((buff80[i]-len(buff80)*0.2)>0)*1) 
buff90=x_test.copy()
buff90=pd.DataFrame(buff90).rank(axis=0)
for i in range(len(buff90.T)):
    buff90[i]=(((buff90[i]-len(buff90)*0.1)>0)*1) 
    
empty=x_test.copy()
empty=pd.DataFrame(empty.std(axis=1))
empty1=x_test.copy()
empty1=pd.DataFrame(empty1.mean(axis=1))
x_test=np.array(pd.concat([buff10,buff20,buff30,buff40,buff50,buff60,buff70,buff80,buff90, empty, empty1], axis=1))
#x_test=np.array(pd.DataFrame(x_test).rank(axis=0)*100/len(x_test))


#%%

# train my model
for epoch in range(training_epochs):
    avg_loss = 0.0
    total_batch = int(len(x_train)/batch_size)
    start = 0
    end = batch_size
    for k in range(total_batch):
        if k != total_batch-1:
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
        else:
            batch_x = x_train[start:]
            batch_y = y_train[start:]
#        buff=batch_x.copy()
#        for i in range(len(buff)):
#            buff[i]=(((buff[i]-buff.mean(axis=0))>0)*1)    
        start += batch_size
        end += batch_size
        loss_val, _ = sess.run([loss, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        avg_loss += loss_val / total_batch
    # Loss of validation set
    cur_val_loss = loss.eval({X: x_valid, Y: y_valid, keep_prob: 1.0})
    print("Epoch: ", str(epoch), "| Train Loss: ", str(avg_loss), "| Valid Loss: ", str(cur_val_loss))

print('Learning Finished!')


# Start testing with test set
##### Check the accuracy of the model - Validate model with the validation set
# Prediction

# Start testing with test set
test_acc = accuracy.eval({X: x_test, Y: y_test, keep_prob: 1.0})
print("Test Accuracy: ", str(test_acc))
#%%
# Test set class returns
class_return = np.zeros([36, 10])  # 뭐라고? (Output 클래스 (정해진 10개 범위 내의 값 반환?)
class_count = np.zeros([36, 10])  # 매월 각각의 클래스에 몇 개의 주식이 할당되는 가를 보기 위한 것

for m in range(36):
    pre_class = prediction.eval({X:x_test[:len(_x_test[m])] ,keep_prob:1.0})
    for cls in range(num_class):
        idx = pre_class == cls
        if any(idx):
            class_return[m][cls] = np.mean(r_test[m].values[pre_class == cls])
            class_count[m][cls] = np.sum([pre_class == cls])

print(pd.DataFrame(class_count))
print(pd.DataFrame(class_return))
r_mean = np.mean(class_return, axis=0)
r_std = np.std(class_return, axis=0)
financial_performance = pd.DataFrame([r_mean, r_std, r_mean/r_std], index=['mean','std','sr'])
print(financial_performance)

class_return[:,:2].mean()-class_return[:,-2:].mean()

empty=pd.DataFrame(class_return)
buff1=empty.iloc[:,:2]
buff2=empty.iloc[:,-2:]
empty=buff1.sum(axis=1)-buff2.sum(axis=1)
empty.mean()
empty.std()
print("평균수익률은 ", empty.mean(), "표준편차는 ", empty.std(), "Sharpe Ratio는" , empty.mean()/empty.std())



