# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:35:51 2015

@author: eeglab
"""

import scipy.io as sio
import lmdb
import numpy as np
import scipy
import h5py
import sys
import os

sys.path.append("/home/eeglab/caffe/python/")
import caffe
caffe.set_mode_gpu()

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd

# Make sure that caffe is on the python path:
caffe_root = '/home/eeglab/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe_root = '/home/eeglab/caffe/'
os.chdir(caffe_root)

#fold = '7';

def produce_hiddenip(fold):
    print('load labels and data')
    data = '/home/eeglab/Downloads/Raw/grp'+fold+'.mat'
    f = h5py.File(data)
    Inputs = f['x_train'][:]
    Labels = f['y_train'][:]
    testInputs = f['x_test'][:]
    testLabels = f['y_test'][:]
    
    Inputs = Inputs*0.1
    testInputs = testInputs*0.1
    
    Inputs = Inputs[:, np.newaxis, :, :]
    testInputs = testInputs[:, np.newaxis, :, :]
    
    print('get train labels and data')
    # training data
    MODEL_FILE = caffe_root+'examples/driving/eeg_train_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_dnn'+fold+'_iter_4000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    out = net.forward(data=Inputs)
    
    ip2 = net.blobs['ip3'].data
    x_train = ip2
    y_train = Labels
    
    print('get test labels and data')
    # testing data
    MODEL_FILE = caffe_root+'examples/driving/eeg_test_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_dnn'+fold+'_iter_4000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    out = net.forward(data=testInputs)
    
    ip2 = net.blobs['ip2'].data
    x_test = ip2
    y_test = testLabels
    
    sio.savemat('/home/eeglab/Downloads/Processed/grp'+fold+'dnn_ip2.mat', \
    	{'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test})
    print('done')

for x in range(1, 8):
    fold = str(x)
    produce_hiddenip(fold)
    

def produce_hiddenip_cnn(fold):
    print('load labels and data')
    data = '/home/eeglab/Downloads/Raw/grp'+fold+'.mat'
    f = h5py.File(data)
    Inputs = f['x_train'][:]
    Labels = f['y_train'][:]
    testInputs = f['x_test'][:]
    testLabels = f['y_test'][:]
    
    Inputs = Inputs*0.1
    testInputs = testInputs*0.1
    
    Inputs = Inputs[:, np.newaxis, :, :]
    testInputs = testInputs[:, np.newaxis, :, :]
    
    print('get train labels and data')
    # training data
    MODEL_FILE = caffe_root+'examples/driving/eeg_train_conv_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_cnn'+fold+'_iter_4000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
#    out = net.forward(data=Inputs)
#    
#    ip2 = net.blobs['ip2'].data
#    x_train = ip2
#    y_train = Labels
    
    input_shape = Inputs.shape
    input_step = int(np.floor(input_shape[0]/3))
    x_train = np.array([],dtype=np.float64).reshape(0,100)
    y_train = []
    for in_idx in range(0, 3):
        in_inputs = Inputs[in_idx*input_step:(in_idx+1)*input_step, :, :, :]
        in_labels = Labels[0, in_idx*input_step:(in_idx+1)*input_step]
        out = net.forward(data=in_inputs)
    
        ip2 = net.blobs['ip2'].data
        x_train = np.concatenate((x_train, ip2), axis=0)
        y_train = np.append(y_train, in_labels)
        
        
    print('get test labels and data')
    # testing data
    MODEL_FILE = caffe_root+'examples/driving/eeg_test_conv_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_cnn'+fold+'_iter_4000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    out = net.forward(data=testInputs)
    
    ip2 = net.blobs['ip2'].data
    x_test = ip2
    y_test = testLabels
    
    sio.savemat('/home/eeglab/Downloads/Processed/grp'+fold+'cnn_ip2.mat', \
    	{'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test})
    print('done')

for x in range(1, 8):
    fold = str(x)
    produce_hiddenip_cnn(fold)
    
    
def produce_hiddenip_ccnn(fold):
    print('load labels and data')
    data = '/home/eeglab/Downloads/Raw/grp'+fold+'.mat'
    f = h5py.File(data)
    Inputs = f['x_train'][:]
    Labels = f['y_train'][:]
    testInputs = f['x_test'][:]
    testLabels = f['y_test'][:]
    
    Inputs = Inputs
    testInputs = testInputs
    
    Inputs = Inputs[:, np.newaxis, :, :]
    testInputs = testInputs[:, np.newaxis, :, :]
    
    print('get train labels and data')
    # training data
    MODEL_FILE = caffe_root+'examples/driving/eeg_train_conv_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_ccnn'+fold+'_iter_4000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
#    out = net.forward(data=Inputs)
#    
#    ip2 = net.blobs['ip2'].data
#    x_train = ip2
#    y_train = Labels
    
    input_shape = Inputs.shape
    input_step = int(np.floor(input_shape[0]/3))
    x_train = np.array([],dtype=np.float64).reshape(0,10, 30, 1)
    y_train = []
    for in_idx in range(0, 3):
        in_inputs = Inputs[in_idx*input_step:(in_idx+1)*input_step, :, :, :]
        in_labels = Labels[0, in_idx*input_step:(in_idx+1)*input_step]
        out = net.forward(data=in_inputs)
    
        ip2 = net.blobs['conv1'].data
        x_train = np.concatenate((x_train, ip2), axis=0)
        y_train = np.append(y_train, in_labels)
        
        
    print('get test labels and data')
    # testing data
    MODEL_FILE = caffe_root+'examples/driving/eeg_test_conv_prod'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_ccnn'+fold+'_iter_6000.caffemodel'
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    out = net.forward(data=testInputs)
    
    ip2 = net.blobs['ip2'].data
    x_test = ip2
    y_test = testLabels
    
    sio.savemat('/home/eeglab/Downloads/Processed/grp'+fold+'ccnn_ip2.mat', \
    	{'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test})
    print('done')

for x in range(1, 8):
    fold = str(x)
    produce_hiddenip_ccnn(fold)
    
#a = out['prob']
#b = a[:, 1]
#fpr, tpr, thresholds = metrics.roc_curve(testLabels[0, :], b, pos_label=1)
#auc_dnn = metrics.auc(fpr,tpr)