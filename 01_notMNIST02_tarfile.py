# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:00:33 2017

@author: hp123
"""
from __future__ import print_function
import numpy as np
import sys
import os
import tarfile

num_classes=10
np.random.seed(133)
data_root="."

def maybe_extract(filename,force=False):
    root=os.path.splitext(os.path.splitext(filename)[0])[0]#splitext将文件名按拓展名划分
    if os.path.isdir(root) and not force:
        print ('%s already present -Skipping extraction os %s' %(root,filename))
    else:
        print ("Extracting data for %s.This may take a wait.Please wait" %root)
        tar=tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders=[\
        os.path.join(root,d) for d in sorted(os.listdir(root)) \
        if os.path.isdir(os.path.join(root,d))]
    if len(data_folders)!=num_classes:
        raise Exception(\
        'Excepted %d folders,one per class.Found %d instead.'%(num_classes,\
        len(data_folders)))
    print (data_folders)
    return data_folders

train_filename = os.path.join(data_root,'notMNIST_large.tar.gz')
test_filename = os.path.join(data_root,'notMNIST_small.tar.gz')

train_folders=maybe_extract(train_filename)
test_folders=maybe_extract(test_filename)