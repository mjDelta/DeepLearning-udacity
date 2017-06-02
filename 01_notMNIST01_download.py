from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display,Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

#matplotlib ÅäÖÃÔÚIPythonÖÐ
#%matplotlib inline

url='http://commondatastorage.googleapis.com/books1000/'
last_percent_reported=None
data_root='.'

def download_progress_hook(count,blockSize,totalSize):
	'''回调函数：汇报下载进度：下载更新每5%汇报一次'''
	global last_percent_reported
	percent=int(count*blockSize*100/totalSize)

	if last_percent_reported!=percent:
		if percent%5==0:
			sys.stdout.write("%s%%" %percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
		last_percent_reported=percent
	
def maybe_download(filename,expected_bytes,force=False):
	'''下载文件，确保是指定大小的文件'''
	dest_filename=os.path.join(data_root,filename)
	if force or not os.path.exists(dest_filename):
		print ("Attempt to download:",filename)
		filename,_=urlretrieve(url+filename,dest_filename,reporthook=download_progress_hook)
		print ('\nDownload Complete!')
	statinfo=os.stat(dest_filename)
	if statinfo.st_size==expected_bytes:
		print ("Found and verified",dest_filename)
	else:
		raise Exception(
			'Failed to verify '+dest_filename+'.Can you get to it with a browser?')
	return dest_filename

train_filename=maybe_download('notMNIST_large.tar.gz',247336696)
test_filename=maybe_download('notMNIST_small.tar.gz',8458043)

train_filename=os.path.join(data_root,'notMNIST_large.tar.gz')
test_filename=os.path.join(data_root,'notMNIST_small.tar.gz')


