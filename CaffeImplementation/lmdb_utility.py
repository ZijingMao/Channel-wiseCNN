# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:46:42 2015

@author: eeglab
"""

import subprocess
import platform
import sys
import os

sys.path.append("/home/eeglab/caffe/python/")
import caffe
caffe.set_mode_gpu()
import lmdb

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/eeglab/caffe/'  # this file is expected to be in {caffe_root}/examples

os.chdir(caffe_root)

lmdb_train_data_name = 'examples/driving/train_driving_enriched_lmdb_1'

lmdb_test_data_name = 'examples/driving/test_driving_enriched_lmdb_1'

print "OS:     ", platform.platform()
print "Python: ", sys.version.split("\n")[0]
print "CUDA:   ", subprocess.Popen(["nvcc","--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])


# Check Content of LMDB
def get_data_for_case_from_lmdb(lmdb_name, id):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()

    raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    feature = caffe.io.datum_to_array(datum)
    label = datum.label

    return (label, feature)
    
# get_data_for_case_from_lmdb(caffe_root+lmdb_train_data_name, "0000001001")

#solver = caffe.get_solver(caffe_root+"examples/driving/eeg_solver1.prototxt")
#solver.solve()

MODEL_FILE = caffe_root+'examples/driving/eeg_train_test_prod1.prototxt'
PRETRAINED = caffe_root+'examples/driving/driving_enriched_dnn1_iter_6000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# get input of the model
Inputs = Inputs[:, np.newaxis, :, :]
net.blobs['data'].data = Inputs

out = net.forward(data=Inputs)