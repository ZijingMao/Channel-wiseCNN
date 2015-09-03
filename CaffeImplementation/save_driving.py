# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:35:33 2015

@author: eeglab
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Make sure that caffe is on the python path:
caffe_root = '/home/eeglab/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.X`

def produce_az_dnn(idx, iteration):
    fold = str(idx)
    MODEL_FILE = caffe_root+'examples/driving/eeg_train_test'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_dnn'+fold+'_iter_'+iteration+'.caffemodel'
    
    
    caffe.set_mode_gpu()
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    
    
    truelbl = net.blobs['label'].data
    out = net.forward()
    prob = out['prob']
    postlbl = prob[:, 1]
    lbl = np.column_stack((postlbl, truelbl))
    
    
    fpr, tpr, thresholds = metrics.roc_curve(truelbl, postlbl, pos_label=1)
    auc_dnn = metrics.auc(fpr,tpr)
    
    return auc_dnn
    
auc_dnn = [0]*7
for x in range(1, 8):
    auc_dnn[x-1] = produce_az_dnn(x, '4000')
np.mean(auc_dnn)
#
#
#np.savetxt("/home/eeglab/caffe/examples/driving/Post_enriched_dnn1.csv", lbl, delimiter=',')


def produce_az_cnn(idx, iteration):
    fold = str(idx)
    MODEL_FILE = caffe_root+'examples/driving/eeg_train_test_conv'+fold+'.prototxt'
    PRETRAINED = caffe_root+'examples/driving/driving_enriched_ccnn'+fold+'_iter_'+iteration+'.caffemodel'
    
    
    caffe.set_mode_cpu()
    
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    
    truelbl = net.blobs['label'].data
    out = net.forward()
    prob = out['prob']
    postlbl = prob[:, 1]
    lbl = np.column_stack((postlbl, truelbl))
    
    fpr, tpr, thresholds = metrics.roc_curve(truelbl, postlbl, pos_label=1)
    auc_cnn = metrics.auc(fpr,tpr)
    
    return auc_cnn
    #np.savetxt("/home/eeglab/caffe/examples/driving/Post_enriched_merged_freq1.csv", lbl, delimiter=',')

auc_ccnn = [0]*7
for x in range(1, 8):
    auc_ccnn[x-1] = produce_az_cnn(x, '6000')
np.mean(auc_ccnn)
   
   
auc_cnn = [0]*5
for x in range(1, 6):
    iterations = str(x*2000)
    auc_cnn[x-1] = produce_az_cnn(7, iterations)    
    
