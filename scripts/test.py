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

        with tf.Session() as sess:

            if osp.isdir(checkpoint):
                checkpoint_path = tf.train.latest_checkpoint(checkpoint)
                if not checkpoint:
                    raise ValueError("No checkpoint file found in: {}".format(checkpoint))

            saver = tf.train.import_meta_graph(checkpoint_path + '.meta')

            saver.restore(sess, checkpoint_path)

            writer = tf.compat.v1.summary.FileWriter(logdir="./training_model", graph=tf.get_default_graph())

if __name__ == '__main__':
    ExportInferenceModel()

