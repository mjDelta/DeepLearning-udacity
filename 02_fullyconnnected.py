# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:27:40 2017

@author: ZMJ
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time
start=time.time()
'''加载数据'''
print("(1)Loading Raw Data...")
pickle_file="notMNIST.pickle"
with open(pickle_file,"rb") as f:
    save=pickle.load(f)
    train_dataset=save["train_dataset"]
    train_labels=save["train_labels"]
    valid_dataset=save["valid_dataset"]
    valid_labels=save["valid_labels"]
    test_dataset=save["test_dataset"]
    test_labels=save["test_labels"]
    '''释放内存'''
    del save
    print("Training set",train_dataset.shape,train_labels.shape)
    print("Validation set",valid_dataset.shape,valid_dataset.shape)
    print("Testing set",test_dataset.shape,test_labels.shape)

image_size=28
num_labels=10

def reformat(dataset,labels):
    '''将三维数据转成二维数据'''
    dataset=dataset.reshape((-1,image_size*image_size)).astype(np.float32)
    '''将label装成多元01label'''
    labels=(np.arange(num_labels)==labels[:,None]).astype(np.float32)
    return dataset,labels
print("(2)Transforming Data shape from 2D to 3D...")    
train_dataset,train_labels=reformat(train_dataset,train_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)

print("Training set :",train_dataset.shape,train_labels.shape)
print("Testing set :",test_dataset.shape,test_labels.shape)
print("Validating set:",valid_dataset.shape,valid_labels.shape)

'''梯度下降'''
print("(3)With GD optimizing...")
train_subset=10000

graph=tf.Graph()
with graph.as_default():
    '''输入数据：向graph加载Train，Test，Valid数据'''
    tf_train_dataset=tf.constant(train_dataset[:train_subset,:])
    tf_train_labels=tf.constant(train_labels[:train_subset,:])
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)
    
    '''变量：需要训练的参数 weights初始化为均值为0，方差为1的正态分布 bias初始化为0'''
    weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biaes=tf.Variable(tf.zeros([num_labels]))
    
    '''训练模型：logistics为预测值 交叉熵作为损失函数'''
    logits=tf.matmul(tf_train_dataset,weights)+biaes
    loss=tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    
    '''优化参数:最小化loss用梯度下降'''
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    '''预测：不是训练模型的一部分 用于检验模型准确率'''
    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biaes)
    test_prediction=tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biaes)
    
'''优化参数'''
num_steps=801
def accuracy(predictions,labels):
    '''np.argmax返回最大值的下标 axis=0列为单位 axis=1行为单位'''
    return (100.*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))\
	/predictions.shape[0])    

with tf.Session(graph=graph) as session:
    '''初始化weights和biaes'''
    tf.initialize_all_variables().run()
    print("Initialized!")
    for step in range(num_steps):
        '''运行optimizer优化参数'''
        _,l,predictions=session.run([optimizer,loss,train_prediction])
        if(step%100==0):
            print("$Loss at step %d:%f" %(step,l))
            print("Trianing accuracy:%.1f%%"\
            % accuracy(predictions,train_labels[:train_subset,:]))
            '''计算valid的预测值'''
            print("Validation accuracy:%.1f%%"\
            % accuracy(valid_prediction.eval(),valid_labels))
    print("Testing accuracy:%.1f%%"%accuracy(test_prediction.eval(),test_labels))
gd_time=time.time()
print("GD running time",gd_time-start)
'''随机梯度下降'''
batch_size=128
print("(4)With SGD optimizing...")
graph=tf.Graph()
with graph.as_default():
    '''输入数据:由于每次优化的样本不一样，用placeholder代替'''
    tf_train_dataset=tf.placeholder(tf.float32,\
									shape=(batch_size,image_size*image_size))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)

    '''参数初始化'''
    weights=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biaes=tf.Variable(tf.zeros([num_labels]))

    '''logits和损失函数'''
    logits=tf.matmul(tf_train_dataset,weights)+biaes
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
						(labels=tf_train_labels,logits=logits))
	
    '''optimizer'''
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    '''training,testing 和 validation的预测值'''
    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biaes)
    test_prediction=tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biaes)

'''开始优化'''
num_steps=3001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized!")
	
    for step in range(num_steps):
        '''随机初始化一批训练集合的下标'''
        offset=(step*batch_size)%(train_labels.shape[0]-batch_size)
        '''生成一批训练数据'''
        batch_dataset=train_dataset[offset:(offset+batch_size),:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        '''用字典将训练数据赋值'''
        feed_dict={tf_train_dataset:batch_dataset,tf_train_labels:batch_labels}
        '''优化参数'''
        _,l,predictions=session.run(\
					[optimizer,loss,train_prediction],feed_dict=feed_dict)
        if(step%500==0):
            print("$Minibatch loss at step %d:%f" %(step,l))
            print("Minibatch accuracy:%.1f%%"%accuracy(predictions,batch_labels))
            print("Validation accuracy:%.1f%%"%accuracy(valid_prediction.eval(),valid_labels))
    print("Testing accuracy:%.1f%%"%accuracy(test_prediction.eval(),test_labels))
sgd_time=time.time()
print("SGD running time:",sgd_time-gd_time)

'''加入RELU函数，增加网络的深度，优化参数使用SGD'''
batch_size=128
hidden_size=1024
print("(5)Adding depth with RELU...")
graph=tf.Graph()
with graph.as_default():
    '''输入数据'''
    tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)
    
    '''初始化参数'''
    W1=tf.Variable(tf.truncated_normal([image_size*image_size,hidden_size]))
    B1=tf.Variable(tf.zeros([hidden_size]))
    
    W2=tf.Variable(tf.truncated_normal([hidden_size,num_labels]))
    B2=tf.Variable(tf.zeros([num_labels]))
    
    '''loss函数初始化'''
    logits1=tf.matmul(tf_train_dataset,W1)+B1
    relus=tf.nn.relu(logits1)
    logits2=tf.matmul(relus,W2)+B2
    
    loss=tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(logits2,tf_train_labels))
    
    '''optimizer初始化'''
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    '''test,valid预测值'''
    predictions_train=tf.nn.softmax(logits2)
    y_valid=tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,W1)+B1),W2)+B2
    predictions_valid=tf.nn.softmax(y_valid)
    y_test=tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,W1)+B1),W2)+B2
    predictions_test=tf.nn.softmax(y_test)
    
num_steps=3001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized...") 
    for step in range(num_steps):
        offset=batch_size*step%(len(train_dataset)-batch_size)
        '''生成一批训练集'''
        batch_data=train_dataset[offset:(offset+batch_size),:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        '''用feed_dict将训练集传入'''
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        '''优化参数开始'''
        _,l,predictions=session.run(\
                    [optimizer,loss,predictions_train],feed_dict=feed_dict)
        
        if (step%500==0):
            print("$MinBatch at step ",step)
            print("Training accuracy:",accuracy(predictions,batch_labels))
            print("Validation accuracy:",accuracy(predictions_valid.eval(),valid_labels))
    print("Testing accuracy:",accuracy(predictions_test.eval(),test_labels))
print("Running time:",time.time()-sgd_time)
    
