#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/driving/eeg_conv_solver1.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver2.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver3.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver4.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver5.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver6.prototxt
./build/tools/caffe train --solver=examples/driving/eeg_conv_solver7.prototxt
