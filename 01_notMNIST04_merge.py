# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:47:59 2017

@author: hp123
"""
from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import os

def make_arrays(nb_rows,img_size):
    if nb_rows:
        dataset=np.ndarray(shape=(nb_rows,img_size,img_size),dtype=np.float32)
        labels=np.ndarray(nb_rows,dtype=np.int32)
    else:
        dataset,labels=None,None
    return dataset,labels

def merge_datasets(pickle_files,train_size,valid_size=0):
    num_classes=len(pickle_files)
    valid_dataset,valid_labels=make_arrays(valid_size,image_size)
    train_dataset,train_labels=make_arrays(train_size,image_size)
    vsize_per_class=valid_size//num_classes
    tsize_per_class=train_size//num_classes
    
    start_v,start_t=0,0
    end_v,end_t=vsize_per_class,tsize_per_class
    end_l=vsize_per_class+tsize_per_class
    for label,pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file) as f:
                letter_set=pickle.load(f)
                '''打乱数据集合顺序'''
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    letter=letter_set[:vsize_per_class,:,:]
                    valid_dataset[start_v:end_v,:,:]=letter
                    valid_labels[start_v:end_v]=label
                    start_v+=vsize_per_class
                    end_v+=vsize_per_class
                letter=letter_set[vsize_per_class:end_l,:,:]
                train_dataset[start_t:end_t,:,:]=letter
                train_labels[start_t:end_t]=label
                start_t+=tsize_per_class
                end_t+=tsize_per_class
        except Exception as e:
            print("Unable to process data from",pickle_file,":",e)
            raise
    return valid_dataset,train_dataset,valid_labels,train_labels

def randomize(dataset,labels):
    permutate=np.random.permutation(labels.shape[0])
    shuffled_dataset=dataset[permutate,:,:]
    shuffled_labels=labels[permutate]
    return shuffled_dataset,shuffled_labels 
                    
train_root=os.path.join(".","notMNIST_large")    
test_root=os.path.join(".","notMNIST_small")

train_files=[os.path.join(train_root,d) for d in os.listdir(train_root)\
            if not os.path.isdir(os.path.join(train_root,d))]
test_files=[os.path.join(test_root,d) for d in os.listdir(test_root) \
            if not os.path.isdir(os.path.join(test_root,d))]

image_size=28
train_size=200000
valid_size=10000
test_size=10000

valid_dataset,train_dataset,valid_labels,train_labels=\
                            merge_datasets(train_files,train_size,valid_size)
_,test_dataset,_,test_labels=merge_datasets(test_files,test_size)

print("Training:",train_dataset.shape,train_labels.shape)
print("Validation:",valid_dataset.shape,valid_labels.shape)
print("Testing:",test_dataset.shape,test_labels.shape)    

train_dataset,train_labels=randomize(train_dataset,train_labels)
valid_dataset,valid_labels=randomize(valid_dataset,valid_labels)
test_dataset,test_labels=randomize(test_dataset,test_labels)
           
pickle_file=os.path.join(".","notMNIST.pickle")
try:
    f=open(pickle_file,"wb")
    save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print("Unable to save data to",pickle_file,":",e)
    raise

statinfo=os.stat(pickle_file)
print("Compressed pickle size:",statinfo.st_size)