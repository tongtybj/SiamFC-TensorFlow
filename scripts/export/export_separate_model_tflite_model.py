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

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '../..'))

from utils.misc_utils import load_cfgs
from siamese_datasets.dataloader import DataLoader

tf.enable_eager_execution()
tensorflow.executing_eagerly()

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib import quantize as contrib_quantize
else:
    import tf_slim as slim
    tf.disable_v2_behavior()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', dest='config_filepath', action="store",
                        help='the path of tracking config for inference', default='Logs/SiamFC/track_model_checkpoints/train', type=str)

    parser.add_argument('--frozen_graph_model', dest='frozen_graph_model', action="store",
                            help='the file path of fronze graph model', default='Logs/SiamFC/track_model_checkpoints/train/models/whole_model_scale1.pb', type=str)


    args, _ = parser.parse_known_args()

    model_save_dir = osp.dirname(args.frozen_graph_model)
    model_config, train_config, track_config = load_cfgs(args.config_filepath)

    train_config['validation_data_config']["batch_size"] = 1
    train_config['validation_data_config']["prefetch_capacity"] = 0 # no need to prefetch
    print("batch.size: {}".format(train_config['validation_data_config'].get("batch_size")))

    tf.enable_eager_execution() # the position of this is very important!!!!, can not be outside

    scale = 1
    with tf.gfile.GFile(args.frozen_graph_model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

      for n in graph_def.node:
        #print(n.name)
        if n.op in ('Placeholder'):
          if n.name == "input_image":
            scale = n.attr['shape'].shape.dim[0].size

    # search image feature extractor
    converter = None
    if model_config['embed_config']['feature_extractor'] == 'alexnet':
        converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['input_image'], ['convolutional_alexnet/conv5/concat'])
    elif model_config['embed_config']['feature_extractor'] == 'mobilenet_v1':
        converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['input_image'], ['MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6'])
    else:
        raise ValueError("incorrecot feature extrator: {}".format(model_config['embed_config']['feature_extractor']))

    tflite_model = converter.convert()
    filename = 'search_image_feature_extractor'
    open(osp.join(model_save_dir, filename + '_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)

    # fully quantized
    def representative_search_data_gen():
      dataloader = DataLoader(train_config['validation_data_config'], False)
      dataloader.build()

      # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
      for _ in range(300):
        exemplar, instance = dataloader.get_one_batch()
        instances = tf.stack([instance[0] for _ in range(scale)])
        yield [instances.numpy().astype(np.float32)]
        #yield [exemplar.numpy()[0].astype(np.float32), instances.numpy().astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_search_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(osp.join(model_save_dir, filename + '_full_quant_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)
    print("finish tflite conversion for {}".format(filename))

    # template image feature extractor
    if model_config['embed_config']['feature_extractor'] == 'alexnet':
      converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['template_image'], ['convolutional_alexnet_1/conv5/concat'])
    elif model_config['embed_config']['feature_extractor'] == 'mobilenet_v1':
      converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['template_image'], ['MobilenetV1_1/MobilenetV1/Conv2d_5_pointwise/Relu6'])

    tflite_model = converter.convert()
    filename = 'template_image_feature_extractor'
    open(osp.join(model_save_dir, filename + '_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)

    # fully quantized
    def representative_template_data_gen():
      dataloader = DataLoader(train_config['validation_data_config'], False)
      dataloader.build()

      # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
      for _ in range(300):
        exemplar, instance = dataloader.get_one_batch()
        yield [exemplar.numpy()[0].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_template_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(osp.join(model_save_dir, filename + '_full_quant_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)
    print("finish tflite conversion for {}".format(filename))

    # corss correlation
    if model_config['embed_config']['feature_extractor'] == 'alexnet':
      converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['convolutional_alexnet_1/conv5/concat', 'convolutional_alexnet/conv5/concat'], ['detection/add'])
    elif model_config['embed_config']['feature_extractor'] == 'mobilenet_v1':
      converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['MobilenetV1_1/MobilenetV1/Conv2d_5_pointwise/Relu6', 'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6'], ['detection/add'])

    tflite_model = converter.convert()
    filename = 'cross_correlation'
    open(osp.join(model_save_dir, filename + '_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)

    # TODO: the cross correlation can not get good quantization performance (accuracy (?) and latency)
    print("finish tflite conversion for {}".format(filename))
