#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Contains definitions of the network in [1].

  [1] Bertinetto, L., et al. (2016).
      "Fully-Convolutional Siamese Networks for Object Tracking."
      arXiv preprint arXiv:1606.09549.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

#import tensorflow as tf
import tensorflow.compat.v1 as tf

from utils.misc_utils import get


TF_MAJOR_VERSION = [ int(num) for num in tf.__version__.split('.')][0]
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib import quantize as contrib_quantize
    from nets import mobilenet_v1
else:
  raise ValueError("not support tensorflow with version 2.x")


def convolutional_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the convolutional_alexnet models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    #print("========= use bn")
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    normalizer_fn = slim.batch_norm
  else:
    batch_norm_params = {}
    normalizer_fn = None

  weight_decay = get(embed_config, 'weight_decay', 5e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', 'kaiming_normal')
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d], # no slim.separable_conv2d
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
        return arg_sc


def convolutional_alexnet(inputs, reuse=None, scope='convolutional_alexnet', split=True, depthwise_list=[]):
  """Defines the feature extractor of SiamFC.

  Args:
    inputs: a Tensor of shape [batch, h, w, c].
    reuse: if the weights in the embedding function are reused.
    scope: the variable scope of the computational graph.

  Returns:
    net: the computed features of the inputs.
    end_points: the intermediate outputs of the embedding function.
  """

  def depthwise_conv(net, end_point_base, kernel, stride, depth):
    end_point = end_point_base + '_depthwise'

    net = slim.separable_conv2d(net, None, kernel,
                                depth_multiplier=1,
                                stride=stride,
                                rate=1,
                                scope=end_point)

    end_point = end_point_base + '_pointwise'

    net = slim.conv2d(net, depth, [1, 1], stride=1, scope=end_point)

    return net

  with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.separable_conv2d],
                        outputs_collections=end_points_collection):
      net = inputs
      if "conv1" in depthwise_list:
        net = depthwise_conv(net, 'conv1', [11, 11], 2, 96)
      else:
        net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

      if "conv2" in depthwise_list:
        net = depthwise_conv(net, 'conv2', [5, 5], 1, 256)
      else:
        if split == True:
          with tf.variable_scope('conv2'):
            b1, b2 = tf.split(net, 2, 3)
            b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
            # https://qiita.com/carushi@github/items/15175cd238f115a51f61
            # The original implementation has bias terms for all convolution, but
            # it actually isn't necessary if the convolution layer is followed by a batch
            # normalization layer since batch norm will subtract the mean.
            b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
            net = tf.concat([b1, b2], 3)
        else:
          net = slim.conv2d(net, 256, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

      if "conv3" in depthwise_list:
        net = depthwise_conv(net, 'conv3', [3, 3], 1, 384)
      else:
        net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')

      if "conv4" in depthwise_list:
        net = depthwise_conv(net, 'conv4', [3, 3], 1, 384)
      else:
        if split == True:
          with tf.variable_scope('conv4'):
            b1, b2 = tf.split(net, 2, 3)
            b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
            b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
            net = tf.concat([b1, b2], 3)
        else:
          net = slim.conv2d(net, 384, [3, 3], 1, scope='conv4')

      # Conv 5 with only convolution, has bias
      if "conv5" in depthwise_list:
        net = depthwise_conv(net, 'conv5', [3, 3], 1, 256)
      else:
        if split == True:
          with tf.variable_scope('conv5'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None, normalizer_fn=None):
              b1, b2 = tf.split(net, 2, 3)
              b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
              b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
              net = tf.concat([b1, b2], 3)
        else:
          net = slim.conv2d(net, 256, [3, 3], 1, scope='conv5')

      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points


convolutional_alexnet.stride = 8
