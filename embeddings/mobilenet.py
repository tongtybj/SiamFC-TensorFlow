from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow.compat.v1 as tf

TF_MAJOR_VERSION = [ int(num) for num in tf.__version__.split('.')][0]
if TF_MAJOR_VERSION == 2:
  raise NameError('Please use tensorflow with version 1.x.x (e.g., 1.15.0), since there is no tensorflow.contrib in version 2.x.x, which is necessary for mobilenet (models/research/slim/nets)')

import tensorflow.contrib.slim as slim
from nets import mobilenet_v1

'''
CONV_DEFS = [
  mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=32),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=64),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=128),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=128),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256)
]
'''
CONV_DEFS = [
  mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=96),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=384),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
  mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256)
]
