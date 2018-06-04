#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy
from scipy.stats import norm
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Download pretrained vgg model
helper.maybe_download_pretrained_vgg(data_dir)


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    print("loading vgg...")
    start_time = time.time()
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    print("--- %s seconds to load vgg ---" % (time.time() - start_time))
         
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    output = tf.layers.conv2d_transpose(vgg_layer7_out, 
                                        num_classes, 4, 2, 
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01) 
    vgg_layer4_out = tf.layers.conv2d(vgg_layer4_out, 
                                      num_classes, 1, strides=(1,1),
                                      padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output, vgg_layer4_out)
    tf.Print(output, [tf.shape(output)])
    
    output = tf.layers.conv2d_transpose(output, 
                                        num_classes, 4, 2, 
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer3_out = tf.layers.conv2d(vgg_layer3_out, 
                                      num_classes, 1, strides=(1,1),
                                      padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)
    output = tf.add(output, vgg_layer3_out)

    output = tf.layers.conv2d_transpose(output, 
                                        num_classes, 16, 8, 
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    regularization_loss =  tf.losses.get_regularization_loss()
    total_loss = cross_entropy_loss + regularization_loss
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    return (logits, train_op, cross_entropy_loss)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """    
    start_time = time.time()
    # TODO: Implement function
    for epoch in range(epochs):
        batch = 0
        for image, lable in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image:image, 
                                          correct_label:lable, 
                                          learning_rate:1e-5, 
                                          keep_prob:0.5})
            batch += 1
        if batch % 20 == 0:
            print(batch, " processed")
            print("Epoch: {} Batch: {} Training Loss: {:.3f}".format(epoch, batch, loss))
            print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        
num_classes = 3
image_shape = (600, 800)
data_dir = './data/combined'
epochs = 5
batch_size = 1
save_model_path = '3_datasets_batchsize1_lr00001_epoch' + str(epochs)

export_dir = './pb_dir_' + save_model_path + '_' + str(int(time.time()))
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.Session() as sess:
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, ''), image_shape)

    # Build NN using load_vgg, layers, and optimize function
    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
    nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
    logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

    sess.run(tf.global_variables_initializer())
    
    # Train NN using the train_nn function
    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
         correct_label, keep_prob, learning_rate)
    
    # Only store trainable variables
    saver_trainable = tf.train.Saver(tf.trainable_variables())
    saver_trainable.save(sess, save_model_path+"_trainable")
    
    # Save variables and pb file
    #saver = tf.train.Saver().save(sess, save_model_path + 'model', global_step = epochs)
    #tf.train.write_graph(sess.graph_def, save_model_path, "graph.pb", as_text = False)



