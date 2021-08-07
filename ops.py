import math
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.framework import ops

from utils import *

'''
try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except BaseException:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter
'''

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.compat.v1.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.compat.v1.concat(tensors, axis, *args, **kwargs)

def bn(x, is_training, scope):
    return tf.compat.v1.layers.batch_normalization(x,
                                        epsilon=1e-5,
                                        scale=True,
                                        name=scope)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.compat.v1.layers.batch_normalization(x,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            name=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.compat.v1.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.compat.v1.nn.conv2d(input_, w, strides=[
                            1, d_h, d_w, 1], padding='SAME')

        biases = tf.compat.v1.get_variable(
            'biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
        conv = tf.compat.v1.reshape(tf.compat.v1.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.compat.v1.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.compat.v1.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.compat.v1.get_variable(
            'biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
        deconv = tf.compat.v1.reshape(tf.compat.v1.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.compat.v1.maximum(x, leak * x)


def linear(input_, output_size,
           scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.compat.v1.float32,
                                           tf.compat.v1.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
                               initializer=tf.compat.v1.constant_initializer(bias_start))
        if with_w:
            return tf.compat.v1.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.compat.v1.matmul(input_, matrix) + bias


def maxpool2d(x, k=2):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    return tf.compat.v1.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# define cross entropy loss
def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            logits=x, labels=y)
    except BaseException:
        return tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            logits=x, targets=y)
