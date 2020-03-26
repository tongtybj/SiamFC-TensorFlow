#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Export inference model from the original trained model with checkpoint"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse

import tensorflow
import tensorflow.compat.v1 as tf

import numpy as np
import cv2

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '../..'))

from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from utils.misc_utils import load_cfgs
from siamese_datasets.dataloader import DataLoader

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
    from nets import mobilenet_v1
    from tensorflow.contrib import quantize as contrib_quantize
else:
    import tf_slim as slim
    tf.disable_v2_behavior()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', action="store",
                            help='the directroy path of traning checkpoint', default='Logs/SiamFC/track_model_checkpoints/train', type=str)

    parser.add_argument('--checkpoint_num', dest='checkpoint_num', action="store",
                            help='the specific checkpoint for export', default=-1, type=int)

    parser.add_argument('--scale', dest='scale', action="store",
                            help='the number of scale to do tracking', default=1, type=int)

    args, _ = parser.parse_known_args()

    model_save_dir = osp.join(args.checkpoint_dir, 'models')
    model_config, train_config, track_config = load_cfgs(args.checkpoint_dir)

    size_z = model_config['z_image_size']
    size_x = track_config['x_image_size']

    tf.enable_eager_execution() # the position of this is very important!!!!, can not be outside

    with tf.Session() as sess:
      template_image = tf.placeholder(tf.float32, shape=[size_z, size_z, 3], name='template_image')
      input_image = tf.placeholder(tf.float32, shape=[args.scale, size_x, size_x, 3], name='input_image')
      template_image =tf.expand_dims(template_image, 0)
      embed_config = model_config['embed_config']

      # build cnn for feature extraction from either template image or input image
      feature_extactor = model_config['embed_config']['feature_extractor']
      if feature_extactor == "alexnet":
        alexnet_config = model_config['alexnet']
        arg_scope = convolutional_alexnet_arg_scope(embed_config,
                                                    trainable=embed_config['train_embedding'],
                                                    is_training=False)
        with slim.arg_scope(arg_scope):
          embed_x, end_points = convolutional_alexnet(input_image, reuse=False, split=alexnet_config['split'])
          embed_z, end_points_z = convolutional_alexnet(template_image, reuse=True, split=alexnet_config['split'])

      elif feature_extactor == "mobilenet_v1":
        mobilenent_config = model_config['mobilenet_v1']
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
          with tf.variable_scope('MobilenetV1', reuse=False) as scope:
            embed_x, end_points = mobilenet_v1.mobilenet_v1_base(input_image, final_endpoint = mobilenent_config['final_endpoint'], depth_multiplier = mobilenent_config['depth_multiplier'], scope=scope)
          with tf.variable_scope('MobilenetV1', reuse=True) as scope:
            embed_z, end_points_z = mobilenet_v1.mobilenet_v1_base(template_image, final_endpoint = mobilenent_config['final_endpoint'], depth_multiplier = mobilenent_config['depth_multiplier'], scope=scope)
      else:
        raise ValueError("Invalid feature extractor: {}".format(feature_extactor))

      # build cross-correlation between features from template image and input image
      with tf.variable_scope('detection'):
        embed_z = tf.squeeze(embed_z, 0)  # [filter_height, filter_width, in_channels]
        embed_z = tf.expand_dims(embed_z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        response = tf.nn.conv2d(embed_x, embed_z, strides=[1, 1, 1, 1], padding='VALID', name="cross_correlation")
        bias = tf.get_variable('biases', [1], dtype=tf.float32, trainable=False)
        response = model_config['adjust_response_config']['scale'] * response + bias

      # upsample the result of cross-correlation
      with tf.variable_scope('upsample'):
        up_method = track_config['upsample_method']
        methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                   'bicubic': tf.image.ResizeMethod.BICUBIC}
        up_method = methods[up_method]
        up_size = [s * track_config['upsample_factor'] for s in response.get_shape().as_list()[1:3]]
        response_up = tf.image.resize(response, up_size, method=up_method, align_corners=True)
        response_up = tf.identity(response_up, name="final_result")

      # saver = tf.train.Saver(tf.global_variables())
      # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage#variables_to_restore
      # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb
      ema = tf.train.ExponentialMovingAverage(0)
      variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])
      saver = tf.train.Saver(variables_to_restore)

      if osp.isdir(args.checkpoint_dir):
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)

        if args.checkpoint_num > 0:
          checkpoint = osp.join(args.checkpoint_dir, "model.ckpt-") + str(args.checkpoint_num)

        if not checkpoint:
          raise ValueError("No checkpoint file found in: {}".format(checkpoint))

      saver.restore(sess, checkpoint)

      #sess.run(tf.initialize_all_variables())

      '''
      for op in tf.get_default_graph().get_operations():
      print(op.name, op.op_def.name)
      for var in tf.global_variables():
      print (var.name)
      #if "convolutional_alexnet/conv1/weights" in var.name:
      if "detection/biases" in var.name:
      print(sess.run(var))
      '''

      # extract frozed graph
      # http://workpiles.com/2016/07/tensorflow-protobuf-dump/
      frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["upsample/final_result"])

      frozen_graph = tf.Graph()
      with frozen_graph.as_default():
        tf.import_graph_def(frozen_graph_def)
        tf.train.write_graph(frozen_graph_def, model_save_dir, 'whole_model_scale' + str(args.scale) + '.pb', as_text=False)

