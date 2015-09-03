# -*- coding: utf-8 -*-
"""
Spyder Editor
Zijing Mao
This is a temporary script file.
"""

import lmdb
import numpy as np
import math
import scipy
import h5py

# Make sure that caffe is on the python path:
caffe_root = '/home/zijing/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Command line to check created files:
# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/train_score_lmdb/

data = '/home/zijing/Downloads/CTXDAWN.mat'

#data = caffe_root+data
#mat = scipy.io.loadmat(data)
#Inputs = mat['x_train']
#Labels = mat['y_train']
#testInputs = mat['x_test']
#testLabels = mat['y_test']

# shuffle the data
#randPos = np.random.permutation(len(Labels))
#Labels = Labels[randPos]
#Inputs = Inputs[:, :, randPos]
#
#Labels = Labels[:, 0]
#testLabels = testLabels[:, 0]

f = h5py.File(data)
Inputs = f['x_train'][:]
Labels = f['y_train'][:]
testInputs = f['x_test'][:]
testLabels = f['y_test'][:]

Labels = Labels[0, :]
testLabels = testLabels[0, :]

randPos = np.random.permutation(len(Labels))
Labels = Labels[randPos]
Inputs = np.transpose(Inputs, (1, 2, 0))
testInputs = np.transpose(testInputs, (1, 2, 0))
Inputs = Inputs[:, :, randPos]

lmdb_train_data_name = 'train_driving_enriched_lmdb_1'

lmdb_test_data_name = 'test_driving_enriched_lmdb_1'

#	entries = re.split(' ', line.strip())
#	Inputs.append(entries[0])
#	Labels.append(entries[1])

print('Writing train labels and data')

# Size of buffer: 1000 elements to reduce memory consumption

for idx in range(int(math.ceil(Inputs.shape[2]/1000.0))):
    tmpInputs = Inputs[:, :, (1000*idx):(1000*(idx+1))]
    tmpLabels = Labels[(1000*idx):(1000*(idx+1))]
    in_db_data = lmdb.open(lmdb_train_data_name, map_size=int(1e12))
    with in_db_data.begin(write=True) as in_txn:
		for in_idx in range(0, tmpInputs.shape[2]):
			im = tmpInputs[:, :, in_idx]
			im = im[np.newaxis, :, :]
			lbl = tmpLabels[in_idx]
			im_dat = caffe.io.array_to_datum(im.astype(float), lbl.astype(int))
			in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())
			string_ = str(1000*idx+in_idx+1) + ' / ' + str(Inputs.shape[2])
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
    in_db_data.close()
print('')


print('Writing test labels and data')

# Size of buffer: 1000 elements to reduce memory consumption
for idx in range(int(math.ceil(testInputs.shape[2]/1000.0))):
    tmpInputs = testInputs[:, :, (1000*idx):(1000*(idx+1))]
    tmpLabels = testLabels[(1000*idx):(1000*(idx+1))]
    in_db_data = lmdb.open(lmdb_test_data_name, map_size=int(1e12))
    with in_db_data.begin(write=True) as in_txn:
		for in_idx in range(0, tmpInputs.shape[2]):
			im = tmpInputs[:, :, in_idx]
			im = im[np.newaxis, :, :]
			lbl = tmpLabels[in_idx]
			im_dat = caffe.io.array_to_datum(im.astype(float), lbl.astype(int))
			in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())
			string_ = str(1000*idx+in_idx+1) + ' / ' + str(testInputs.shape[2])
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
    in_db_data.close()
print('')

print('Done')