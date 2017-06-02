# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:44:22 2017

@author: ZMJ
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file="notMNIST.pickle"

'''加载数据'''
print("(1)Loading data...")
with open(pickle_file) as f:
    save=pickle.load(f)
    train_dataset=save["train_dataset"]
    train_labels=save["train_labels"]
    valid_dataset=save["valid_dataset"]
    valid_labels=save["valid_labels"]
    test_dataset=save["test_dataset"]
    test_labels=save["test_labels"]
    
    '''释放内存'''
    del save
    
    print("\tTraining set ",train_dataset.shape,train_labels.shape)
    print("\tValidation set ",valid_dataset.shape,valid_labels.shape)
    print("\tTesting set ",test_dataset.shape,test_labels.shape)

'''改变数据维数：
    ①data：3D到2D
    ②labels：one-hot编码'''
image_size=28
num_labels=10
print("(2)Reformating dataset and labels...")
def reformat(dataset,labels):
    dataset=dataset.reshape((-1,image_size*image_size).astype(np.float32))
    '''one-hot编码转换'''
    labels=(np.arange(num_labels)==labels[:None]).astype(np.float32)
    return dataset,labels

train_dataset,train_labels=reformat(train_dataset,train_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)
print("\tTraining set ",train_dataset.shape,train_labels.shape)
print("\tValidation set ",valid_dataset.shape,valid_labels.shape)
print("\tTesting set ",test_dataset.shape,test_labels.shape)

regular_param=0.0001
'''定义准确率'''
def accuracy(predictions,labels):
    return 100.*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0]

'''L2 regularization'''
print("(3)L2 regularization with SGD and Relus...")
batch_size=128
hidden_size=512

graph=tf.Graph()
with graph.as_default():
    '''数据'''
    tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)
    
    '''变量'''
    w1=tf.Variable(tf.truncated_normal([image_size*image_size,hidden_size]))
    b1=tf.Variable(tf.zeros([hidden_size]))
    
    w2=tf.Variable(tf.truncated_normal([hidden_size,num_labels]))
    b2=tf.Variable(tf.zeros([num_labels]))
    
    '''loss L2'''
    y=tf.matmul(tf_train_dataset,w1)+b1
    relus=tf.nn.rule(y)
  
	 '''dropout规则,tf.nn.dropout(x,probility),probility为激活值被随机删除的概率'''
	 #relus=tf.nn.dropout(relus,0.8)
    logits=tf.matmul(relus,w2)+b2

    loss=tf.reduce_mean(tf.softmax_cross_entropy_with_logits(logits,tf_train_labels))\
									+regular_param*tf.nn.l2_loss(w1)+regular_param*tf.nn.l2_loss(w2)
    
    '''optimizer'''
    optimizer=tf.train.GradientDescentOpyimizer(0.5).minimize(loss)
    
    '''test和valid预测值'''
    train_predictions=tf.nn.softmax(logits)
    valid_y=tf.matmul(tf_valid_dataset,w1)+b1
    valid_predictions=tf.nn.relu(y,w2)+b2
    test_y=tf.matmul(tf_test_dataset,w1)+b1
    test_predictions=tf.matmul(tf.nn.rule(test_y),w2)+b2

num_steps=3001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("\tInitialized!")
    for step in range(num_steps):
        offset=step*batch_size%(train_dataset.shape[0]-batch_size)
		'''构造过拟合数据集'''
		'''
		if(offset>10000)offset=0
		'''
        batch_data=train_dataset[offset:(offset+batch_size),:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,l,predictions=session.run(\
                [optimizer,loss,train_predictions],feed_dict=feed_dict)
        if(step%500==0):
            print("\tStep ",step,":Loss is ",l)
            print("\tTraining set accuracy :",accuracy(predictions,batch_labels))
            print("\tValidation set accuracy :",accuracy(valid_predictions,valid_labels))
    print("\Testing set accuracy :",accuracy(test_predictions,test_labels))