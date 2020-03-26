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

    parser.add_argument('--frozen_graph_model', dest='frozen_graph_model', action="store",
                            help='the file path of fronze graph model', default='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained/models/whole_model_scale1.pb', type=str)

    parser.add_argument('--config', dest='config_filepath', action="store",
                        help='the path of tracking config for inference', default='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained', type=str)

    args, _ = parser.parse_known_args()

    model_save_dir = osp.dirname(args.frozen_graph_model)
    model_config, train_config, track_config = load_cfgs(args.config_filepath)

    tf.enable_eager_execution() # the position of this is very important!!!!, can not be outside

    scale = 1
    with tf.gfile.GFile(args.frozen_graph_model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

      for n in graph_def.node:
        if n.op in ('Placeholder'):
          if n.name == "input_image":
            scale = n.attr['shape'].shape.dim[0].size

    # we do not add the last upsample operation in tensorflow since this should be implemented in customized operation. Too much cost, and this is easy to implement in CPU process.
    converter = tf.lite.TFLiteConverter.from_frozen_graph(args.frozen_graph_model, ['template_image', 'input_image'], ['detection/add'])

    '''
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter#from_frozen_graph
    converter.allow_custom_ops = True
    '''
    tflite_model = converter.convert()
    filename, ext = osp.splitext(osp.basename(args.frozen_graph_model))
    open(osp.join(model_save_dir, filename + '.tflite'), "wb").write(tflite_model)

    # quantization: https://arxiv.org/pdf/1712.05877.pdf
    # only weigh quatization is very slow x5
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #tflite_model = converter.convert()

    # load dataset for representative data for activation quantization
    train_config['validation_data_config']["batch_size"] = 1
    train_config['validation_data_config']["prefetch_capacity"] = 0 # no need to prefetch
    print("batch.size: {}".format(train_config['validation_data_config'].get("batch_size")))

    # fully quantized
    def representative_data_gen():
      dataloader = DataLoader(train_config['validation_data_config'], False)
      dataloader.build()

      # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
      for _ in range(300):
        exemplar, instance = dataloader.get_one_batch()
        instances = tf.stack([instance[0] for _ in range(scale)])
        yield [exemplar.numpy()[0].astype(np.float32), instances.numpy().astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(osp.join(model_save_dir, 'whole_model_full_quant_scale' + str(scale) + '.tflite'), "wb").write(tflite_model)
