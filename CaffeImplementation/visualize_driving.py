# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:46:03 2015

@author: eeglab
"""

from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from caffe.proto import caffe_pb2
from IPython.display import display, Image 

caffe_root = '/home/eeglab/caffe/'  # this file is expected to be in {caffe_root}/examples
MODEL_FILE = caffe_root+'examples/driving/eeg_train_test.prototxt'
PRETRAINED = caffe_root+'examples/driving/driving_enriched1_iter_10000.caffemodel'

_net = caffe_pb2.NetParameter()
f = open(MODEL_FILE)
text_format.Merge(f.read(), _net)
display(Image(get_pydot_graph(_net,"TB").create_png()))