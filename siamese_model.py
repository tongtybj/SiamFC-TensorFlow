#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Construct the computational graph of siamese model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v1 as tf

from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from metrics.track_metrics import center_dist_error, center_score_error
from utils.train_utils import construct_gt_score_maps, load_mat_model
from siamese_datasets.dataloader import DataLoader

TF_MAJOR_VERSION = [ int(num) for num in tf.__version__.split('.')][0]
if TF_MAJOR_VERSION == 1:
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib import quantize as contrib_quantize
    from nets import mobilenet_v1
else:
    import tf_slim as slim


class SiameseModel:
  def __init__(self, model_config, train_config, track_config, mode='train'):
    self.model_config = model_config
    self.train_config = train_config
    self.track_config = track_config
    self.mode = mode
    assert mode in ['train', 'validation']

    if self.mode == 'train':
      self.data_config = self.train_config['train_data_config']
    elif self.mode == 'validation':
      self.data_config = self.train_config['validation_data_config']

    self.dataloader = None
    self.exemplars = None
    self.instances = None
    self.response = None
    self.batch_loss = None
    self.total_loss = None
    self.init_fn = None
    self.global_step = None

    if self.train_config["export"]:
      self.data_config['batch_size'] = 1

  def is_training(self):
    """Returns true if the model is built for training mode"""
    return self.mode == 'train'

  def build_inputs(self):
    """Input fetching and batching

    Outputs:
      self.exemplars: image batch of shape [batch, hz, wz, 3]
      self.instances: image batch of shape [batch, hx, wx, 3]
    """
    if self.train_config["export"]:
      size_z = self.model_config['z_image_size']
      size_x = self.track_config['x_image_size']

      '''
      template_image = tf.placeholder(shape=[1, size_z, size_z, 3],
                                          dtype=tf.float32,
                                          name='template_image')
      input_image = tf.placeholder(shape=[1, size_x, size_x, 3],
                                          dtype=tf.float32,
                                          name='input_image')
      self.exemplars =tf.expand_dims(template_image, 0)
      self.instances =tf.expand_dims(input_image, 0)
      '''
      self.exemplars = tf.placeholder(shape=[1, size_z, size_z, 3],
                                      dtype=tf.float32,
                                      name='template_image')
      self.instances = tf.placeholder(shape=[1, size_x, size_x, 3],
                                      dtype=tf.float32,
                                      name='input_image')

    else:
      with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
        self.dataloader = DataLoader(self.data_config, self.is_training())
        self.dataloader.build()
        exemplars, instances = self.dataloader.get_one_batch()

        self.exemplars = tf.to_float(exemplars)
        self.instances = tf.to_float(instances)

  def build_image_embeddings(self, reuse):
    """Builds the image model subgraph and generates image embeddings

    Inputs:
      self.exemplars: A tensor of shape [batch, hz, wz, 3]
      self.instances: A tensor of shape [batch, hx, wx, 3]

    Outputs:
      self.exemplar_embeds: A Tensor of shape [batch, hz_embed, wz_embed, embed_dim]
      self.instance_embeds: A Tensor of shape [batch, hx_embed, wx_embed, embed_dim]
    """
    feature_extactor = self.model_config['embed_config']['feature_extractor']

    feature_extractors = {'alexnet': self.build_image_embeddings_alexnet,
                          'mobilenet_v1': self.build_image_embeddings_mobilenet_v1}

    return feature_extractors[feature_extactor](reuse)

  def build_image_embeddings_alexnet(self, reuse=False):
    model_config = self.model_config['embed_config']
    alexnet_config = self.model_config['alexnet']
    arg_scope = convolutional_alexnet_arg_scope(model_config,
                                                trainable=model_config['train_embedding'],
                                                is_training=self.is_training())

    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):

        with slim.arg_scope(arg_scope):
            return convolutional_alexnet(images, reuse=reuse, split=alexnet_config['split'])

    self.exemplar_embeds, _ = embedding_fn(self.exemplars, reuse=reuse)
    self.instance_embeds, _ = embedding_fn(self.instances, reuse=True)

  def build_image_embeddings_mobilenet_v1(self, reuse=False):
    """Builds the image model subgraph and generates image embeddings based on mobilenent
    """
    model_config = self.model_config['embed_config']
    mobilenent_config = self.model_config['mobilenet_v1']

    def embedding_fn(images, reuse=False):
        with tf.variable_scope('MobilenetV1', reuse=reuse) as scope:
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
                return mobilenet_v1.mobilenet_v1_base(images,
                                                      final_endpoint = mobilenent_config['final_endpoint'],
                                                      depth_multiplier = mobilenent_config['depth_multiplier'], scope=scope)

    self.exemplar_embeds, _ = embedding_fn(self.exemplars, reuse=reuse)
    self.instance_embeds, _ = embedding_fn(self.instances, reuse=True)

  def build_template(self):
    # The template is simply the feature of the exemplar image in SiamFC.
    self.templates = self.exemplar_embeds

  def build_detection(self, reuse=False):
    with tf.variable_scope('detection', reuse=reuse):
      def _translation_match(x, z):  # translation match for one example within a batch
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds, self.templates),
                         dtype=self.instance_embeds.dtype)
      output = tf.squeeze(output, [1, 4])  # of shape e.g., [8, 15, 15]

      # Adjust score, this is required to make training possible.
      config = self.model_config['adjust_response_config']
      bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=config['train_bias'])
      response = config['scale'] * output + bias
      self.response = response

  def build_loss(self):
    response = self.response
    response_size = response.get_shape().as_list()[1:3]  # [height, width]

    gt = construct_gt_score_maps(response_size,
                                 self.data_config['batch_size'],
                                 self.model_config['embed_config']['stride'],
                                 self.train_config['gt_config'])

    # loss: https://www.renom.jp/ja/notebooks/tutorial/basic_algorithm/lossfunction/notebook.html
    with tf.name_scope('Loss'):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=response,
                                                     labels=gt)

      with tf.name_scope('Balance_weights'):
        n_pos = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 1)))
        n_neg = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 0)))
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        class_weights = tf.where(tf.equal(gt, 1),
                                 w_pos * tf.ones_like(gt),
                                 tf.ones_like(gt))
        class_weights = tf.where(tf.equal(gt, 0),
                                 w_neg * tf.ones_like(gt),
                                 class_weights)
        loss = loss * class_weights

      # Note that we use reduce_sum instead of reduce_mean since the loss has
      # already been normalized by class_weights in spatial dimension.
      loss = tf.reduce_sum(loss, [1, 2])

      batch_loss = tf.reduce_mean(loss, name='batch_loss')
      tf.losses.add_loss(batch_loss)

      total_loss = tf.losses.get_total_loss()
      self.batch_loss = batch_loss
      self.total_loss = total_loss

      # quantization
      # good note: https://www.tensorflowers.cn/t/7136
      if self.model_config['embed_config']['quantization']:
        if self.train_config["export"]:
          contrib_quantize.create_eval_graph()
        else:
          contrib_quantize.create_training_graph(quant_delay=200000)

      tf.summary.image('exemplar', self.exemplars, family=self.mode)
      tf.summary.image('instance', self.instances, family=self.mode)

      mean_batch_loss, update_op1 = tf.metrics.mean(batch_loss)
      mean_total_loss, update_op2 = tf.metrics.mean(total_loss)
      with tf.control_dependencies([update_op1, update_op2]):
        tf.summary.scalar('batch_loss', mean_batch_loss, family=self.mode)
        tf.summary.scalar('total_loss', mean_total_loss, family=self.mode)

      if self.mode == 'train':
        tf.summary.image('GT', tf.reshape(gt[0], [1] + response_size + [1]), family='GT')
      tf.summary.image('Response', tf.expand_dims(tf.sigmoid(response), -1), family=self.mode)
      tf.summary.histogram('Response', self.response, family=self.mode)

      # Two more metrics to monitor the performance of training
      tf.summary.scalar('center_score_error', center_score_error(response), family=self.mode)
      tf.summary.scalar('center_dist_error', center_dist_error(response), family=self.mode)

  def setup_global_step(self):
    global_step = tf.Variable(
      initial_value=0,
      name='global_step',
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def setup_embedding_initializer(self):
    """Sets up the function to restore embedding variables from checkpoint."""
    embed_config = self.model_config['embed_config']
    if embed_config['embedding_checkpoint_file']:
      # Restore Siamese FC models from .mat model files
      initialize = load_mat_model(embed_config['embedding_checkpoint_file'],
                                  'convolutional_alexnet/', 'detection/')

      def restore_fn(sess):
        tf.warn.info("Restoring embedding variables from checkpoint file %s",
                        embed_config['embedding_checkpoint_file'])
        sess.run([initialize])

      self.init_fn = restore_fn

  def build(self, reuse=False):
    """Creates all ops for training and evaluation"""
    with tf.name_scope(self.mode):
      self.build_inputs()
      self.build_image_embeddings(reuse=reuse)
      self.build_template()
      self.build_detection(reuse=reuse)
      self.setup_embedding_initializer()

      if self.mode in ['train', 'validation']:
        self.build_loss()

      if self.is_training():
        self.setup_global_step()
