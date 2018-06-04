#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
pip install scikit-video
wget -nc https://s3.amazonaws.com/lyft-udacity/3_datasets_batchsize1_lr00001_epoch5_trainable.meta -P /tmp/
wget -nc https://s3.amazonaws.com/lyft-udacity/3_datasets_batchsize1_lr00001_epoch5_trainable.index -P /tmp/
wget -nc https://s3.amazonaws.com/lyft-udacity/3_datasets_batchsize1_lr00001_epoch5_trainable.data-00000-of-00001 -P /tmp/
wget -nc https://s3.amazonaws.com/lyft-udacity/checkpoint -P /tmp/
pip install opencv-python

# Add your desired packages for each workspace initialization
#          Add here!          #
