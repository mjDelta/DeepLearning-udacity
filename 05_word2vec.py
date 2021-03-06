# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:15:20 2017

@author: ZMJ
"""
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
#from sklearn.manifold import TSNE

url="http://mattmahoney.net/dc/"

def maybe_download(filename,expected_bytes):
    '''若路径下不存在该文件，就下载，并核对下载的字节数'''
    if not os.path.exists(filename):
        filename,_=urlretrieve(url+filename,filename)
    statinfo=os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print("① Found and verified ",filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Failed to verify ",filename,".Download with a broser!")
    return filename

filename=maybe_download("text8.zip",31344016)

def read_data(filename):
    '''解压文件，提取第一个文件的内容'''
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words=read_data(filename)
print("② Data size is ",len(words))

'''建立数据集'''
vocabulary_size=50000

def build_dataset(words):
    count=[["UNK",-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            idx=dictionary[word]
        else:
            idx=0#不常出现的单词下标
            unk_count+=1
        data.append(idx)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary
data,count,dictionary,reverse_dictionary=build_dataset(words)
print("③ Most common words(including UNK) ",count[:5])
print("\tSample data ",data[:10]) 
del words

data_index=0
'''skip-gram的数据'''
def generate_batch(batch_size,num_skips,skip_window):
	global data_index
	assert batch_size%num_skips==0
	assert num_skips<=2*skip_window
	batch=np.ndarray(shape=(batch_size),dtype=np.int32)
	labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
	span=2*skip_window+1
	buffer=collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index=(data_index+1)%len(data)
	for i in range(batch_size//num_skips):
		target=skip_window
		targets_to_avoid=[skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target=random.randint(0,span-1)
			targets_to_avoid.append(target)
			batch[i*num_skips+j]=buffer[skip_window]
			labels[i*num_skips+j]=buffer[target]
		buffer.append(data[data_index])
		data_index=(data_index+1)%len(data)
	return batch,labels

print("data:",[reverse_dictionary[di] for di in data[:8]])


batch_size=128
embedding_size=128
skip_window=1
num_skips=2
valid_size=16
valid_window=100
valid_examples=np.array(random.sample(range(valid_window),valid_size))
num_sampled=64

graph=tf.Graph()

with graph.as_default(),tf.device("/cpu:0"):

    train_dataset=tf.placeholder(tf.int32,shape=[batch_size])
    train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    

    embeddings=tf.Variable(\
                tf.random_uniform([vocabulary_size,embedding_size],-1.,1.))
    softmax_weights=tf.Variable(\
                tf.truncated_normal([vocabulary_size,embedding_size],\
                stddev=1.0/math.sqrt(embedding_size)))
    softmax_biases=tf.Variable(tf.zeros([vocabulary_size]))
    

    embed=tf.nn.embedding_lookup(embeddings,train_dataset)
    loss=tf.reduce_mean(\
            tf.nn.sampled_softmax_loss(weights=softmax_weights,biases=softmax_biases,\
                    inputs=embed,labels=train_labels,num_sampled=num_sampled,\
                    num_classes=vocabulary_size))


    optimizer=tf.train.AdagradOptimizer(1.0).minimize(loss)
    

    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings=embeddings/norm
    valid_embeddings=tf.nn.embedding_lookup(\
                normalized_embeddings,valid_dataset)
    similarity=tf.matmul(valid_embeddings,tf.transpose(normalized_embeddings))
    
num_steps=100001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized!")
    average_loss=0
    for step in range(num_steps):
        batch_data,batch_labels=generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_dataset:batch_data,train_labels:batch_labels}
        _,l=session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss+=l
        if step%2000==0:
            if step>0:
                average_loss=average_loss/2000.
            print("Average loss at step %d:%f" %(step,average_loss))
            average_loss=0
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=reverse_dictionary[valid_examples[i]]
                top_k=8
                nearest=(-sim[i,:]).argsort()[1:top_k+1]
                log="Nearest to %s: "%valid_word
                for k in range(top_k):
                    close_word=reverse_dictionary[nearest[k]]
                    log="%s %s"%(log,close_word)
                print(log)
    final_embeddings=normalized_embeddings.eval()
                    
