__author__ = 'eeglab'
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
MODEL_FILE = caffe_root+'examples/driving/eeg_train_test_conv2.prototxt'
PRETRAINED = caffe_root+'examples/driving/driving_ccnn2_iter_4000.caffemodel'


caffe.set_mode_cpu()
# net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# ip4 = net.blobs['ip3'].data

out = net.forward()

prob = out['prob']
prob = prob[:, 0:2]

post = prob[:, 1]

fpr, tpr, thresholds = metrics.roc_curve(testLabels, post, pos_label=1)
auc = metrics.auc(fpr,tpr)


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s