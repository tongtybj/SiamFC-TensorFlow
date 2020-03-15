#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Export inference model from the original trained model with checkpoint"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp

import tensorflow as tf
#from tensorflow.python.framework import graph_util

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))


from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from utils.misc_utils import load_cfgs

checkpoint = 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'
input_files = 'assets/KiteSurf'

slim = tf.contrib.slim

class ExportInferenceModel():
    def __init__(self):
        self.model_config, _, self.track_config = load_cfgs(checkpoint)
        size_z = self.model_config['z_image_size']
        size_x = self.track_config['x_image_size']

        with tf.Session() as sess:
            template_image = tf.compat.v1.placeholder(tf.float32, shape=[size_z, size_z, 3], name='template_image')
            input_image = tf.compat.v1.placeholder(tf.float32, shape=[size_x, size_x, 3], name='input_image')
            template_image =tf.expand_dims(template_image, 0)
            input_image =tf.expand_dims(input_image, 0)
            embed_config = self.model_config['embed_config']

            # build cnn for feature extraction from either template image or input image
            arg_scope = convolutional_alexnet_arg_scope(embed_config,
                                                    trainable=embed_config['train_embedding'],
                                                    is_training=False)
            with slim.arg_scope(arg_scope):
                embed_x, end_points = convolutional_alexnet(input_image, reuse=False)
                embed_z, end_points_z = convolutional_alexnet(template_image, reuse=True)
                #print("input image output: {}".format(embed_x.shape))
                #print("template image output: {}".format(embed_z.shape))


            # build cross-correlation between features from template image and input image
            with tf.compat.v1.variable_scope('detection'):
                embed_z = tf.squeeze(embed_z, 0)  # [filter_height, filter_width, in_channels]
                embed_z = tf.expand_dims(embed_z, -1)  # [filter_height, filter_width, in_channels, out_channels]
                response = tf.nn.conv2d(embed_x, embed_z, strides=[1, 1, 1, 1], padding='VALID', name="cross_correlation")
                response = tf.squeeze(response)  # of shape [17, 17]
                bias = tf.compat.v1.get_variable('biases', [1], dtype=tf.float32, trainable=False)
                response = self.model_config['adjust_response_config']['scale'] * response + bias

            # upsample the result of cross-correlation
            with tf.compat.v1.variable_scope('upsample'):
                up_method = self.track_config['upsample_method']
                methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                           'bicubic': tf.image.ResizeMethod.BICUBIC}
                up_method = methods[up_method]
                up_size = [s * self.track_config['upsample_factor'] for s in response.get_shape().as_list()]
                response = tf.expand_dims(response, -1)
                response_up = tf.image.resize(response, up_size, method=up_method, align_corners=True)
                response_up = tf.squeeze(response_up, name="final_result")

            #print("final op: {}".format(response_up.name))

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            if osp.isdir(checkpoint):
                checkpoint_path = tf.train.latest_checkpoint(checkpoint)
                if not checkpoint:
                    raise ValueError("No checkpoint file found in: {}".format(checkpoint))

            saver.restore(sess, checkpoint_path)

            #sess.run(tf.initialize_all_variables())
            #writer = tf.summary.FileWriter(logdir="./inference_model", graph=tf.get_default_graph())
            ''' unncessary debug part'''
            '''
            variables_to_restore = []
            for op in tf.get_default_graph().get_operations():
                print(op.name, op.op_def.name)
                if op.op_def and 'Variable' in op.op_def.name:
                    variables_to_restore.append(op.name)

            for var in tf.global_variables():
                print (var.name)
                #if "convolutional_alexnet/conv1/weights" in var.name:
                if "detection/biases" in var.name:
                    print(sess.run(var))
            '''

            # extract frozed graph
            # http://workpiles.com/2016/07/tensorflow-protobuf-dump/
            # frozen_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["convolutional_alexnet/conv5/concat", "convolutional_alexnet_1/conv5/concat"])
            frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, tf.compat.v1.get_default_graph().as_graph_def(), ["upsample/final_result"])

            frozen_graph = tf.Graph()
            with frozen_graph.as_default():
                tf.import_graph_def(frozen_graph_def)
                #for op in frozen_graph.get_operations():
                    #if op.name == "import/convolutional_alexnet/conv1/weights":
                        #print (sess.run(op))

                writer = tf.compat.v1.summary.FileWriter(logdir="./inference_model", graph=frozen_graph)
                tf.train.write_graph(frozen_graph_def, './inference_model', 'frozen_graph.pb', as_text=False)

if __name__ == '__main__':
    ExportInferenceModel()

