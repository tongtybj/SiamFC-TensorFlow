#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import random
import time
from datetime import datetime

import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver

import configuration
import siamese_model
from utils.misc_utils import auto_select_gpu, mkdir_p, have_cfgs, save_cfgs, load_cfgs

ex = Experiment(configuration.RUN_NAME)
ex.observers.append(FileStorageObserver.create(osp.join(configuration.LOG_DIR, 'sacred')))


### bakui memo:
### 1. learning rate algorithm
### 2. optimizer algorithm

@ex.config
def configurations():
  # Add configurations for current script, for more details please see the documentation of `sacred`.
  # REFER: http://sacred.readthedocs.io/en/latest/index.html
  model_config = configuration.MODEL_CONFIG
  train_config = configuration.TRAIN_CONFIG
  track_config = configuration.TRACK_CONFIG


def _configure_learning_rate(train_config, global_step):
  lr_config = train_config['lr_config']

  num_batches_per_epoch = \
    int(train_config['train_data_config']['num_examples_per_epoch'] / train_config['train_data_config']['batch_size'])

  lr_policy = lr_config['policy']
  if lr_policy == 'piecewise_constant':
    lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    return tf.train.piecewise_constant(global_step,
                                       lr_boundaries,
                                       lr_config['lr_values'])
  elif lr_policy == 'exponential':
    decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
    return tf.train.exponential_decay(lr_config['initial_lr'],
                                      global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_config['lr_decay_factor'],
                                      staircase=lr_config['staircase'])
  elif lr_policy == 'cosine':
    T_total = train_config['train_data_config']['epoch'] * num_batches_per_epoch
    return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.to_float(global_step) / T_total))
  else:
    raise ValueError('Learning rate policy [%s] was not recognized', lr_policy)


def _configure_optimizer(train_config, learning_rate):
  optimizer_config = train_config['optimizer_config']
  optimizer_name = optimizer_config['optimizer'].upper()
  if optimizer_name == 'MOMENTUM':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=optimizer_config['momentum'],
      use_nesterov=optimizer_config['use_nesterov'],
      name='Momentum')
  elif optimizer_name == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
  return optimizer


@ex.automain
def main(model_config, train_config, track_config):
  os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

  # Create training directory which will be used to save: configurations, model files, TensorBoard logs
  train_dir = train_config['train_dir']
  if not osp.isdir(train_dir):
    logging.info('Creating training directory: %s', train_dir)
    mkdir_p(train_dir)

  if have_cfgs(train_dir):
    model_config, train_config, track_config = load_cfgs(train_dir)
    print("=================== load cfg ")
  else:
    save_cfgs(train_dir, model_config, train_config, track_config)
    print("=================== save default cfg, please modify files in {}".format(train_dir))
    return

  g = tf.Graph()
  with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    # Build the training and validation model
    model = siamese_model.SiameseModel(model_config, train_config, track_config, mode='train')
    model.build()
    model_va = siamese_model.SiameseModel(model_config, train_config, track_config, mode='validation')
    model_va.build(reuse=True)

    learning_rate = _configure_learning_rate(train_config, model.global_step)
    optimizer = _configure_optimizer(train_config, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # general way for run train: https://qiita.com/horiem/items/00ec6488b23895cc4fe2
    # tensorflow 2.1: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    # Set up the training ops
    opt_op = tensorflow.contrib.layers.optimize_loss(
      loss=model.total_loss,
      global_step=model.global_step,
      learning_rate=learning_rate,
      optimizer=optimizer,
      clip_gradients=train_config['clip_gradients'],
      learning_rate_decay_fn=None,
      summaries=['learning_rate'])

    with tf.control_dependencies([opt_op]):
      train_op = tf.no_op(name='train')

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    summary_writer = tf.summary.FileWriter(train_dir, g)
    summary_op = tf.summary.merge_all()

    global_variables_init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()
    g.finalize()  # Finalize graph to avoid adding ops by mistake

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])

    if not model_path:
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      start_step = 0

      if model_config['embed_config']['embedding_checkpoint_file']:
        model.init_fn(sess)
    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(local_variables_init_op)
      saver.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1

    # export
    if train_config["export"]:
      # still debugging
      '''
      frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["train/detection/add"])
      frozen_graph = tf.Graph()
      with frozen_graph.as_default():
        tf.import_graph_def(frozen_graph_def)
        save_model_dir = osp.join(train_config['train_dir'], 'models')
        tf.train.write_graph(frozen_graph_def, save_model_dir, 'quantized_frozen_graph.pb', as_text=False)
        tf.train.write_graph(frozen_graph_def, save_model_dir, 'quantized_frozen_graph.pbtxt', as_text=True)

        output_op = sess.graph.get_tensor_by_name("validation/detection/add:0")
        input1_op = sess.graph.get_tensor_by_name("validation/template_image:0")
        input2_op = sess.graph.get_tensor_by_name("validation/input_image:0")

        converter = tf.lite.TFLiteConverter.from_session(sess, [input1_op, input2_op], [output_op])
        converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0] : (0., 1.), input_arrays[1] : (0., 1.)}  # mean, std_dev
        converter.default_ranges_stats = (0, 255)
        tflite_model = converter.convert()
        open(osp.join(save_model_dir, 'quantized_frozen_graph.tflite'), "wb").write(tflite_model)
      '''
      return

    # Training loop
    data_config = train_config['train_data_config']
    total_steps = int(data_config['epoch'] *
                      data_config['num_examples_per_epoch'] /
                      data_config['batch_size'])
    logging.info('Train for {} steps'.format(total_steps))
    save_step = int(data_config['num_examples_per_epoch'] / data_config['batch_size'])
    print("=========== save_step: {}".format(save_step))
    for step in range(start_step, total_steps):
      start_time = time.time()
      # no "feed_dict"
      # has "feed_dict" exmaple (mnist): https://qiita.com/SwitchBlade/items/6677c283b2402d060cd0
      _, loss, batch_loss, instances, response = sess.run([train_op, model.total_loss, model.batch_loss, model.instances, model.response])
      duration = time.time() - start_time
      if step % 10 == 0:
        examples_per_sec = data_config['batch_size'] / float(duration)
        time_remain = data_config['batch_size'] * (total_steps - step) / examples_per_sec
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        format_str = ('%s: step %d, total loss = %.2f, batch loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch; %dh:%02dm:%02ds remains)')
        logging.info(format_str % (datetime.now(), step, loss, batch_loss,
                                   examples_per_sec, duration, h, m, s))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % save_step == 0 or (step + 1) == total_steps:
        checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
