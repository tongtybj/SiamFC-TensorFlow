#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

## bakui memo:
## 1. data agumentation algorithm
## 2. data batch algorithm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import itertools

import tensorflow.compat.v1 as tf

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])

from siamese_datasets.sampler import Sampler
from siamese_datasets.transforms import Compose, RandomGray, RandomCrop, CenterCrop, RandomStretch
from siamese_datasets.vid import VID
from utils.misc_utils import get

class DataLoader(object):
  def __init__(self, config, is_training):
    self.config = config
    self.is_training = is_training

    preprocess_name = get(config, 'preprocessing_name', None)
    logging.info('preproces -- {}'.format(preprocess_name))

    if preprocess_name == 'siamese_fc_color':
      self.v_transform = None
      # TODO: use a single operation (tf.image.crop_and_resize) to achieve all transformations ?
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'siamese_fc_gray':
      self.v_transform = RandomGray()
      self.z_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)), # embeding stride: 8, robustness: should also be center
                                  RandomCrop(255 - 2 * 8),
                                  CenterCrop((127, 127))])
      self.x_transform = Compose([RandomStretch(),
                                  CenterCrop((255 - 8, 255 - 8)),
                                  RandomCrop(255 - 2 * 8), ])
    elif preprocess_name == 'None':
      self.v_transform = None
      self.z_transform = CenterCrop((127, 127))
      self.x_transform = CenterCrop((255, 255))
    else:
      raise ValueError('Preprocessing name {} was not recognized.'.format(preprocess_name))

    self.dataset_py = VID(config['input_imdb'], config['max_frame_dist'])
    self.sampler = Sampler(self.dataset_py, shuffle=is_training)

  def build(self):
    self.build_dataset()
    self.build_iterator()

  def transform_fn(self, video):
    exemplar_file = tf.read_file(video[0])
    instance_file = tf.read_file(video[1])
    exemplar_image = tf.image.decode_jpeg(exemplar_file, channels=3, dct_method="INTEGER_ACCURATE")
    instance_image = tf.image.decode_jpeg(instance_file, channels=3, dct_method="INTEGER_ACCURATE")

    if self.v_transform is not None:
      video = tf.stack([exemplar_image, instance_image])
      video = self.v_transform(video)
      exemplar_image = video[0]
      instance_image = video[1]

    if self.z_transform is not None:
      exemplar_image = self.z_transform(exemplar_image)

    if self.x_transform is not None:
      instance_image = self.x_transform(instance_image)

    return exemplar_image, instance_image

  def build_dataset(self):
    def sample_generator():
      for video_id in self.sampler:
        sample = self.dataset_py[video_id]
        yield sample

    # https://qiita.com/S-aiueo32/items/c7e86ef6c339dfb013ba
    # TODO: except tf.errors.OutOfRangeError: # 末尾まで行ったらループを抜ける
    dataset = tf.data.Dataset.from_generator(sample_generator,
                                             output_types=(tf.string),
                                             output_shapes=(tf.TensorShape([2])))
    # prefecth, thread: http://tensorflow.classcat.com/2019/03/23/tf20-alpha-guide-data-performance/
    # https://www.tensorflow.org/tutorials/load_data/images?hl=ja
    dataset = dataset.map(self.transform_fn, num_parallel_calls=self.config['prefetch_threads'])
    dataset = dataset.prefetch(self.config['prefetch_capacity'])
    dataset = dataset.repeat()
    dataset = dataset.batch(self.config['batch_size'])
    print(" ======= batch size: {}".format(self.config['batch_size']))
    self.dataset_tf = dataset

  def build_iterator(self):
      self.iterator = self.dataset_tf.make_one_shot_iterator()

  def get_one_batch(self):
    return self.iterator.get_next()

  def build_eager_dataset(self):

    def sample_generator():
      for video_id in self.sampler:
      #for video_id in itertools.count(1):
        print("===== smaple generator")
        sample = self.dataset_py[video_id]
        print(sample)
        exemplar_image, instance_image = self.transform_fn(sample)
        print(exemplar_image.shape)
        print(instance_image.shape)

        yield exemplar_image, instance_image

    self.dataset_tf = tf.data.Dataset.from_generator(sample_generator,
                                                     output_types=(tf.string),
                                                     output_shapes=(tf.TensorShape([2])))

    return self.dataset_tf
