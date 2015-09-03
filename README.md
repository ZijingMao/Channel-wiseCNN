Channel-wise CNN (CCNN)
==========================================

This code is for the Cognitive test (poor and good performance)
for [NCTU lane-keeping driving test](http://sccn.ucsd.edu/~jung/pdf/IEEECAS05.pdf)

## matlab implementation
In this code we used stacked restricted Boltzmann machine (RBMs) as convolution filter
Running Az7Fold.m for 7-fold cross-subject test
Running Az10FOld.m for 10-fold within-subject test

## caffe implementation
In this code we used traditional convolution filter but in channel-wise
run train_eeg_conv.sh & train_eeg_conv_batch.sh after [caffe](http://caffe.berkeleyvision.org/) installation

./produce_output.py script will generate output probability and area under the curve (AUC)

Used in the paper:

Feature extraction with deep belief networks for driver’s cognitive states prediction from EEG Data

Novel Deep Neural Network Based Models and Machine Learning Methods For Driver’s Cognitive State Prediction From EEG Signals

Prediction of the Drowsy and Alert States of the Drivers with Novel Deep Neural Network Method from EEG Signals
