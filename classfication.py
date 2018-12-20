# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:29:06 2018

@author: 우람
"""

tf.reset_default_graph()

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose
 
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss
 
        return False

from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import pickle
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\알고')

tf.get_default_session()


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

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

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    return cell



x_train, y_train, r_train, x_test, y_test, r_test= get_sample()




#%%
sess=tf.Session()
sess.run(tf.global_variables_initializer())

timestpes=1 #시퀀스길이랑 같음 
seq_length = 1 #7일 동안 있으니까 sequence length도 7개
data_dim = 5 #input size...  open, high, low, volume, close 이렇게 5개
hidden_dim = 10
output_dim = 1
learning_rate = 0.1
iterations = 500
nb_classes=10

# Scale each
#train_set = MinMaxScaler(train_set)
#test_set = MinMaxScaler(test_set)

#%%
sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 


keep_prob=tf.placeholder(tf.float32)
X=tf.placeholder(tf.float32, [None, 11])
Y=tf.placeholder(tf.int32, [None, 1])   
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
 
W1=tf.get_variable("W1", shape=[11, 64], initializer=tf.contrib.layers.xavier_initializer())
b1= tf.Variable(tf.random_normal([64]))
L1= tf.nn.relu(tf.matmul(X,W1)+b1)
L1= tf.nn.dropout(L1, keep_prob=keep_prob)

W2=tf.get_variable("W2", shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b2= tf.Variable(tf.random_normal([32]))
L2= tf.nn.relu(tf.matmul(L1,W2)+b2)
L2= tf.nn.dropout(L2, keep_prob=keep_prob)

W3=tf.get_variable("W3", shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
b3= tf.Variable(tf.random_normal([16]))
L3= tf.nn.relu(tf.matmul(L2,W3)+b3)
L3= tf.nn.dropout(L3, keep_prob=keep_prob)

W4=tf.get_variable("W4", shape=[16, 16], initializer=tf.contrib.layers.xavier_initializer())
b4= tf.Variable(tf.random_normal([16]))
L4= tf.nn.relu(tf.matmul(L3,W4)+b4)
L4= tf.nn.dropout(L4, keep_prob=keep_prob)


W5=tf.get_variable("W5", shape=[16, 10], initializer=tf.contrib.layers.xavier_initializer())
b5= tf.Variable(tf.random_normal([10]))
hypothesis= tf.matmul(L4,W5)+b5
#hypothesis=tf.arg_max(hypothesis,1)

#loss = tf.reduce_sum(tf.square(hypothesis - Y))
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer=tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost) 
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost) 

predicton = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(predicton, tf.arg_max(Y_one_hot,1))
#predicton = hypothesis 
#is_correct= tf.equal(predicton, Y)
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))


#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#
#    # Training step
#    early_stopping = EarlyStopping(patience=4, verbose=1)
#    for epoch in range(10):
#        for size in range(36):
#            avg_cost=0
#            for i in range(iterations):
#                Xs=np.array(x_train[size])
#                Ys=np.float64(np.array(y_train[size]))
#                Xs=Xs.reshape([len(Xs),data_dim])
#                Ys=np.float64(Ys.reshape([len(Xs),seq_length]))
#                _, step_loss = sess.run([optimizer, cost], feed_dict={X:Xs, Y: Ys, keep_prob:0.6})
#                avg_cost += step_loss/500
#                print("[step: {}] loss: {}".format(i, step_loss))
#                print(avg_cost)
#        Xt=np.array(x_test[size])
#        Xt=Xt.reshape([len(Xt), data_dim])
#        Yt=np.array(y_test[size])
#        Yt=Yt.reshape([len(Yt),seq_length])        
#        val_cost = cost.eval(session=sess, feed_dict={X:Xt, Y: Yt,keep_prob: 1.0})
#        if early_stopping.validate(val_cost):
#            break
##    Xt=np.array(x_test[size])
##    Xt=Xt.reshape([len(Xt), data_dim])
##    Yt=np.array(y_test[size])
##    Yt=Yt.reshape([len(Yt),seq_length])
#        print("Accuracy", accuracy.eval(session=sess, feed_dict={X:Xt, Y:Yt, keep_prob:1})) #여긴 1
#        print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:Xt, keep_prob:1})) #여기도 1!!
#    

#tf.reset_default_graph()

#%%        
#        
#        print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={Xt})) #여기도 1!!
#        Xt=np.array(x_test[size])
#        Xt=Xt.reshape([len(Xt), data_dim])
#        Yt=np.array(y_test[size])
#        Yt=Yt.reshape([len(Yt),seq_length])
#        test_predict = sess.run(prediction, feed_dict={X: Xt})
#        rmse_val = sess.run(rmse, feed_dict={targets: Yt, predictions: test_predict})
#        print("RMSE: {}".format(rmse_val))

iterations=2
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    early_stopping = EarlyStopping(patience=3, verbose=1)
    for epoch in range(10):
        for size in range(36):
#            x_tra = np.concatenate([v for k, v in x_train.items() if k < size])
#            y_tra = np.concatenate([v for k, v in y_train.items() if k < size])
            x_tra=x_train[size]
            y_tra=y_train[size]
            buff=x_tra.copy()
            buff.columns=range(0,5)
            for i in range(len(x_tra.T)):
                buff[i]=(((buff[i]-buff.mean(axis=0)[i])>0)*1)        
            empty=np.array(pd.concat([pd.DataFrame(x_tra), pd.DataFrame(x_tra+1).prod(axis=1)-1, buff, pd.DataFrame(y_tra)], axis=1))
            x_tra=np.array(pd.DataFrame(shuffle(empty)).iloc[:,:-1])
            y_tra=np.array(pd.DataFrame(shuffle(empty)).iloc[:,-1])
            y_tra=np.float64(y_tra.reshape([len(y_tra),1]))
            avg_cost=0
            for i in range(iterations):
#                Xs=np.array(x_train[size])
#                Ys=np.float64(np.array(y_train[size]))
#                x_tra=Xs.reshape([len(X_tra),data_])
#                y_tra_one_hot=tf.one_hot(y_tra, nb_classes)
#                y_tra_one_hot=tf.reshape(y_tra_one_hot,[-1,nb_classes])
                _, step_loss = sess.run([optimizer, cost], feed_dict={X:x_tra, Y: y_tra, keep_prob:0.6})
                avg_cost += step_loss/iterations
                print("[step: {}] loss: {}".format(i, step_loss))
                print(avg_cost)
#        x_tes = np.concatenate([v for k, v in x_test.items() if k < 36])
            x_tes=x_test[size]
            buff=x_tes.copy()
            buff.columns=range(0,5)
            for i in range(len(x_tes.T)):
                buff[i]=((buff[i]-buff.mean(axis=0)[i])>0)*1     
            x_tes=np.array(pd.concat([pd.DataFrame(x_tes), pd.DataFrame(x_tes+1).prod(axis=1)-1, buff], axis=1))
#            y_tes = np.concatenate([v for k, v in y_test.items() if k < 36])
            y_tes=y_test[size]
            y_tes=np.float64(y_tes.reshape([len(y_tes),seq_length]))
#        Xt=np.array(x_test[size])
#        Xt=Xt.reshape([len(Xt), data_dim])
#        Yt=np.array(y_test[size])
#        Yt=Yt.reshape([len(Yt),seq_length])        
            val_cost = cost.eval(session=sess, feed_dict={X:x_tes, Y: y_tes,keep_prob: 1.0})

#    Xt=np.array(x_test[size])
#    Xt=Xt.reshape([len(Xt), data_dim])
#    Yt=np.array(y_test[size])
#    Yt=Yt.reshape([len(Yt),seq_length])
            print("Accuracy", accuracy.eval(session=sess, feed_dict={X:x_tes, Y:y_tes, keep_prob:1})) #여긴 1
            print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:x_tes, keep_prob:1})) #여기도 1!!
            a=sess.run(tf.arg_max(hypothesis,1), feed_dict={X:x_tes, keep_prob:1})
            if early_stopping.validate(val_cost):
                break

#sess.close()
tf.reset_default_graph()









