#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, skvideo.io, json, base64, time, warnings
import os.path
from io import BytesIO, StringIO
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy
from scipy.stats import norm
import scipy.misc
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLASSES = 3
CUT_TOP = 128
CUT_BOTTOM = 88
IMAGE_SHAPE = (600-CUT_TOP-CUT_BOTTOM, 800)
CAR_THRETHOOD = 0.45
ROAD_THRETHOOD = 0.6

# Define encoder function
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

file = sys.argv[-1]
video = skvideo.io.vread(file)

answer_key = {}
# Frame numbering starts at 1
frame = 1
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    start = time.time()
    loader = tf.train.import_meta_graph("/tmp/3_datasets_batchsize1_lr00001_epoch5_trainable.meta")
    loader.restore(sess, tf.train.latest_checkpoint('/tmp/'))

    loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    input_image = loaded_graph.get_tensor_by_name('image_input:0')

    #print(0, time.time() - start)
    start = time.time()
    for rgb_frame in video:
        image_cut = rgb_frame[CUT_TOP:-CUT_BOTTOM,:]
        #print(frame, "sess begin..")
        start = time.time()
        im_softmax = sess.run(
            [tf.nn.softmax(loaded_logits)],
            {keep_prob: 1.0, input_image: [image_cut]})
        
        #print(frame, "sess cost:", time.time() - start)
        start = time.time()
        
        softmax_results = im_softmax[0].reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES)
        
        binary_road_result = np.zeros((600, 800)).astype('uint8')#.astype('uint8')
        interest_area = binary_road_result[CUT_TOP:-CUT_BOTTOM]
        np.copyto(interest_area, (softmax_results[:, :, 1] > ROAD_THRETHOOD).astype('uint8'))
        
        binary_car_result = np.zeros((600, 800)).astype('uint8')#.astype('uint8')
        interest_area = binary_car_result[CUT_TOP:-CUT_BOTTOM]
        np.copyto(interest_area, (softmax_results[:, :, 2] > CAR_THRETHOOD).astype('uint8'))
        
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        #print(frame, time.time() - start)
        start = time.time()
        # Increment frame
        frame += 1    
        
# Print output in proper json format
print (json.dumps(answer_key))