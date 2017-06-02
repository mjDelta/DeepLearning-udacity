# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:07:03 2017

@author: ZMJ
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

image_size=28
num_channels=1
num_labels=10

def reformat(dataset,labels):
    dataset=dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
    labels=(np.arange(num_labels)==labels[:,None]).astype(np.float32)
    return dataset,labels

def accuracy(predictions,labels):
    return 100.*np.sum(np.argmax(labels,1)==np.argmax(predictions,1))/predictions.shape[0] 
    
'''载入数据：序列化'''
print("① Loading Data...")
pickle_file="notMNIST.pickle"
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
    print("\tTraining shape:",train_dataset.shape,train_labels.shape)
    print("\tValidation shape:",valid_dataset.shape,valid_labels.shape)
    print("\tTesting shape:",test_dataset.shape,test_labels.shape)

'''数据输入维度转换'''
print("② Reformating Data...")
train_dataset,train_labels=reformat(train_dataset,train_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)
print("\tTraining shape",train_dataset.shape,train_labels.shape)
print("\tValidation shape",valid_dataset.shape,valid_labels.shape)
print("\tTesting shape",test_dataset.shape,test_labels.shape)

'''随机梯度下降优化'''
batch_size=128
patch_size=5
depth=16
num_hidden=64
'''定义运行graph'''
graph=tf.Graph()
with graph.as_default():
    #输入数据
    tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,\
                                    num_channels))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)
    
    #定义权重和偏置项
    '''结构为：前两层为隐藏层（每层都是卷积，激活函数为relu）'''
    global_step=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(0.08,global_step,200,0.98)
    w1=tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev=0.1))
    b1=tf.Variable(tf.zeros([depth]))
    w2=tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1))
    b2=tf.Variable(tf.constant(1.0,shape=[depth]))
    w3=tf.Variable(tf.truncated_normal([4*4*depth,num_hidden],stddev=0.1))
    b3=tf.Variable(tf.constant(1.0,shape=[num_hidden]))
    w4=tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))    
    b4=tf.Variable(tf.constant(1.0,shape=[num_labels]))
    
    '''定义模型函数'''
    def model(data):
        conv=tf.nn.conv2d(data,w1,[1,2,2,1],padding="SAME")
	conv=tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding="SAME")
        hidden=tf.nn.relu(conv+b1)
	hidden=tf.nn.dropout(hidden,0.5)
	conv=tf.nn.conv2d(conv,w2,[1,2,2,1],padding="SAME")
        hidden=tf.nn.relu(conv+b2)
	hidden=tf.nn.dropout(hidden,0.5)
        shape=hidden.get_shape().as_list()
        reshape=tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
        hidden=tf.nn.relu(tf.matmul(reshape,w3)+b3)
        return tf.matmul(hidden,w4)+b4
    
    '''训练数据值定义'''
    logits=model(tf_train_dataset)
    loss=tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))

    '''定义优化器'''
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    '''train,valid,test预测值'''
    train_predictions=tf.nn.softmax(logits)
    valid_logits=model(tf_valid_dataset)
    valid_predictions=tf.nn.softmax(valid_logits)
    test_logits=model(tf_test_dataset)
    test_predictions=tf.nn.softmax(test_logits)

print("③ Constructing CNN...")
num_steps=10001    
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        offset=(step*batch_size)%(train_dataset.shape[0]-batch_size)
        batch_data=train_dataset[offset:(offset+batch_size),:,:,:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,l,predictions=session.run([optimizer,loss,train_predictions],\
		feed_dict=feed_dict)
        if(step%500==0):
            print("Step:",step)
            print("\tTraining set accuracy:",accuracy(predictions,batch_labels))
            print("\tValidation set accuracy:",accuracy(valid_predictions.eval(),valid_labels))
    print("\tTesting set accuracy:",accuracy(test_predictions.eval(),test_labels))
            
        
