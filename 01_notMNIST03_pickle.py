# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:17:19 2017

@author: hp123
"""
from __future__ import print_function
import os
from scipy import ndimage
import numpy as np
from six.moves import cPickle as pickle

image_size=28   #定义图片像素px，宽度高度
pixel_depth=255

def load_letter(folder,min_num_images):
    '''将一类字母转成数组'''
    image_files=os.listdir(folder)
    dataset=np.ndarray(shape=(len(image_files),image_size,image_size),\
                       dtype=np.float32)
    print (folder)
    num_images=0
    for image in image_files:
        image_file=os.path.join(folder,image)
        try:
            image_data=(ndimage.imread(image_file).astype(float)-\
                        pixel_depth/2)/pixel_depth
            if image_data.shape!=(image_size,image_size):
                raise Exception("Unexpected image shape: %s" %str(image_data.shape))
            dataset[num_images,:,:]=image_data
            num_images=num_images+1
        except IOError as e:
            print ("Couldn't read: ",image_file,":",e,"-it's ok,skipping.")
    dataset=dataset[:num_images,:,:]
    if num_images<min_num_images:
        raise Exception('Many fewer images than expected:%d<%d'%(num_images,min_num_images))
        
    print ("Full dataset tensor:",dataset.shape)
    print ("Mean:",np.mean(dataset))
    print ("Standard deviation:",np.std(dataset))
    return dataset

def maybe_pickle(data_folders,min_num_images_per_class,force=False):
    '''将数组持久化到pickle中'''
    dataset_names=[]
    for folder in data_folders:
        set_filename=folder+".pickle"
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print ("%s already present-Skipping pickling."%set_filename)
        else:
            print ("Pickling %s."%set_filename)
            dataset=load_letter(folder,min_num_images_per_class)
            try:
                with open(set_filename,"wb") as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print ("Unable to save data to",set_filename,":",e)
    return dataset_names
train_filenames=os.path.join(".","notMNIST_large")
test_filenames=os.path.join(".","notMNIST_small")
train_folders=[os.path.join(train_filenames,d) for d in os.listdir(train_filenames)\
                if os.path.isdir(os.path.join(train_filenames,d))]
test_folders=[os.path.join(test_filenames,d) for d in os.listdir(test_filenames)\
                if os.path.isdir(os.path.join(test_filenames,d))]
train_datasets=maybe_pickle(train_folders,45000)
test_datasets=maybe_pickle(test_folders,1800)