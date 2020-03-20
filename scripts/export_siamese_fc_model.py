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
sys.path.append(osp.join(CURRENT_DIR, '..'))

from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from utils.misc_utils import load_cfgs
from datasets.dataloader import DataLoader

#tensorflow.enable_eager_execution()

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
else:
    import tf_slim as slim
    tf.disable_v2_behavior()


class ExportInferenceModel():
    def __init__(self, checktpoint_dir):
        self.checkpoint_dir = checktpoint_dir
        self.model_save_dir = osp.join(self.checkpoint_dir, 'models')
        self.model_config, self.train_config, self.track_config = load_cfgs(self.checkpoint_dir)

        self.dataloader = DataLoader(self.train_config['validation_data_config'], False)
        self.dataloader.build()


        print("===== split: {}".format(self.model_config['embed_config']["split"]))
        size_z = self.model_config['z_image_size']
        size_x = self.track_config['x_image_size']

        # load dataset for representative data for activation quantization
        self.train_config['validation_data_config']["batch_size"] = 1
        self.train_config['validation_data_config']["prefetch_capacity"] = 0 # no need to prefetch
        print("batch.size: {}".format(self.train_config['validation_data_config'].get("batch_size")))
        with tf.Session() as sess:
            template_image = tf.placeholder(tf.float32, shape=[size_z, size_z, 3], name='template_image')
            input_image = tf.placeholder(tf.float32, shape=[size_x, size_x, 3], name='input_image')
            template_image =tf.expand_dims(template_image, 0)
            input_image =tf.expand_dims(input_image, 0)
            embed_config = self.model_config['embed_config']

            # build cnn for feature extraction from either template image or input image
            arg_scope = convolutional_alexnet_arg_scope(embed_config,
                                                    trainable=embed_config['train_embedding'],
                                                    is_training=False)
            with slim.arg_scope(arg_scope):
                embed_x, end_points = convolutional_alexnet(input_image, reuse=False, split=embed_config['split'])
                embed_z, end_points_z = convolutional_alexnet(template_image, reuse=True, split=embed_config['split'])
                #print("input image output: {}".format(embed_x.shape))
                #print("template image output: {}".format(embed_z.shape))


            # build cross-correlation between features from template image and input image
            with tf.variable_scope('detection'):
                embed_z = tf.squeeze(embed_z, 0)  # [filter_height, filter_width, in_channels]
                embed_z = tf.expand_dims(embed_z, -1)  # [filter_height, filter_width, in_channels, out_channels]
                response = tf.nn.conv2d(embed_x, embed_z, strides=[1, 1, 1, 1], padding='VALID', name="cross_correlation")
                response = tf.squeeze(response)  # of shape [17, 17]
                bias = tf.get_variable('biases', [1], dtype=tf.float32, trainable=False)
                response = self.model_config['adjust_response_config']['scale'] * response + bias

            # upsample the result of cross-correlation
            with tf.variable_scope('upsample'):
                up_method = self.track_config['upsample_method']
                methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                           'bicubic': tf.image.ResizeMethod.BICUBIC}
                up_method = methods[up_method]
                up_size = [s * self.track_config['upsample_factor'] for s in response.get_shape().as_list()]
                response = tf.expand_dims(response, -1)
                response_up = tf.image.resize(response, up_size, method=up_method, align_corners=True)
                response_up = tf.squeeze(response_up, name="final_result")

            #print("final op: {}".format(response_up.name))

            saver = tf.train.Saver(tf.global_variables())
            if osp.isdir(self.checkpoint_dir):
                checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if not checkpoint:
                    raise ValueError("No checkpoint file found in: {}".format(checkpoint))

            saver.restore(sess, checkpoint)

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
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["upsample/final_result"])

            frozen_graph = tf.Graph()
            with frozen_graph.as_default():
                tf.import_graph_def(frozen_graph_def)
                writer = tf.summary.FileWriter(logdir=self.model_save_dir, graph=frozen_graph)
                tf.train.write_graph(frozen_graph_def, self.model_save_dir, 'frozen_graph.pb', as_text=False)

                # tensorflow lite

                for op in frozen_graph.get_operations():
                    print(op.name)

                # we do not add the last upsample operation in tensorflow since this should be implemented in customized operation. Too much cost, and this is easy to implement in CPU process.
                converter = tf.lite.TFLiteConverter.from_frozen_graph(osp.join(self.model_save_dir, 'frozen_graph.pb'), ['template_image', 'input_image'], ['detection/add'])

                '''
                # https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter#from_frozen_graph
                converter.allow_custom_ops = True
                '''
                converter.dump_graphviz_dir = './inference_model'
                tflite_model = converter.convert()
                open(osp.join(self.model_save_dir, 'converted_model.tflite'), "wb").write(tflite_model)

                # quantization: https://arxiv.org/pdf/1712.05877.pdf
                # only weigh quatization is very slow x5
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                tflite_model = converter.convert()
                open(osp.join(self.model_save_dir, 'converted_model_weight_quant.tflite'), "wb").write(tflite_model)

                # fully quantized
                #tf.executing_eagerly()

                def representative_data_gen():
                    for i in range(300):
                        exemplar_op, instance_op = self.dataloader.get_one_batch()
                        exemplar, instance = sess.run([exemplar_op, instance_op])
                        #exemplar, instance = self.dataloader.get_one_batch()

                        yield [exemplar[0].astype(np.float32), instance[0].astype(np.float32)]

                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                tflite_model = converter.convert()
                open(osp.join(self.model_save_dir, 'converted_model_full_quant.tflite'), "wb").write(tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', action="store",
                            help='the directroy path of traning checkpoint', default='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained', type=str)

    args, _ = parser.parse_known_args()

    ExportInferenceModel(args.checkpoint_dir)

