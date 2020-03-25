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
from siamese_datasets.dataloader import DataLoader


TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
    from nets import mobilenet_v1
    from tensorflow.contrib import quantize as contrib_quantize
else:
    import tf_slim as slim
    tf.disable_v2_behavior()


class ExportInferenceModel():
    def __init__(self, checktpoint_dir, checkpoint_num):
        self.checkpoint_dir = checktpoint_dir
        self.model_save_dir = osp.join(self.checkpoint_dir, 'models')
        self.model_config, self.train_config, self.track_config = load_cfgs(self.checkpoint_dir)

        tf.enable_eager_execution() # the position of this is very important!!!!, can not be outside

        size_z = self.model_config['z_image_size']
        size_x = self.track_config['x_image_size']

        # load dataset for representative data for activation quantization
        self.train_config['validation_data_config']["batch_size"] = 1
        self.train_config['validation_data_config']["prefetch_capacity"] = 0 # no need to prefetch

        # tensorflow lite
        converter = tf.lite.TFLiteConverter.from_frozen_graph(osp.join(self.model_save_dir, 'frozen_graph.pb'), ['convolutional_alexnet_1/conv5/concat', 'convolutional_alexnet/conv5/concat'], ['detection/add'])
        # converter = tf.lite.TFLiteConverter.from_frozen_graph(osp.join(self.model_save_dir, 'frozen_graph_cross_correlation.pb'), ['embed_z', 'embed_x'], ['detection/add'])

        tflite_model = converter.convert()
        open(osp.join(self.model_save_dir, 'converted_model_cross_correlation.tflite'), "wb").write(tflite_model)

        def representative_data_gen():
          dataloader = DataLoader(self.train_config['validation_data_config'], False)
          dataloader.build()

          graph = tf.Graph()

          with tf.gfile.GFile(osp.join(self.model_save_dir, 'frozen_graph.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

          with graph.as_default():
            tf.import_graph_def(graph_def)
            graph.finalize()

          sess = tf.Session(graph = graph)

          for _ in range(int(self.train_config['train_data_config']['num_examples_per_epoch']/2)):
            exemplar, instance = dataloader.get_one_batch()

            template_op = graph.get_tensor_by_name("import/convolutional_alexnet_1/conv5/concat:0")
            search_op = graph.get_tensor_by_name("import/convolutional_alexnet/conv5/concat:0")

            embed_z, embed_x = sess.run([template_op, search_op], feed_dict = {"import/template_image:0": exemplar.numpy()[0], "import/input_image:0": instance.numpy()[0]})

            yield [embed_z, embed_x]

        converter.representative_dataset = representative_data_gen
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        open(osp.join(self.model_save_dir, 'converted_model_full_quant_cross_correlation.tflite'), "wb").write(tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', action="store",
                            help='the directroy path of traning checkpoint', default='Logs/SiamFC/track_model_checkpoints/train', type=str)

    parser.add_argument('--checkpoint_num', dest='checkpoint_num', action="store",
                            help='the directroy path of traning checkpoint', default=-1, type=int)

    args, _ = parser.parse_known_args()

    tf.enable_eager_execution()

    ExportInferenceModel(args.checkpoint_dir, args.checkpoint_num)

