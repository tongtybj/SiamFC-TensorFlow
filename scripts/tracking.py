#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
from glob import glob
import re
import argparse
import collections
import tensorflow.compat.v1 as tf
#import tensorflow as tf
import cv2
import json
import numpy as np
import datetime

import tflite_runtime.interpreter as tflite

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from utils.misc_utils import sort_nicely

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

def get_center(x):
    return (x - 1.) / 2.

def convert_bbox_format(bbox, to):
  x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
  if to == 'top-left-based':
      x -= get_center(target_width)
      y -= get_center(target_height)
  elif to == 'center-based':
      y += get_center(target_height)
      x += get_center(target_width)
  else:
      raise ValueError("Bbox format: {} was not recognized".format(to))

  return Rectangle(x, y, target_width, target_height)

class TargetState(object):
  """Represent the target state."""

  def __init__(self, bbox, search_pos, scale_idx):
    self.bbox = bbox  # (cx, cy, w, h) in the original image
    self.search_pos = search_pos  # target center position in the search image
    self.scale_idx = scale_idx  # scale index in the searched scales

class SiameseTracking():
  def __init__(self, config_filepath, separate_mode, whole_model, template_model, search_model, cross_model):

    self.separate_mode = separate_mode
    self.whole_lite = False
    self.template_image_quant = False
    self.search_image_quant = False
    self.num_scales = 1

    # whole model
    if not separate_mode:
      if ".pb" in whole_model:
        print("load frozen graph model")
        self.graph = tf.Graph()
        with tf.gfile.GFile(whole_model, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())

        for n in graph_def.node:
          if n.op in ('Placeholder'):
            if n.name == "input_image":
              self.num_scales = n.attr['shape'].shape.dim[0].size

        with self.graph.as_default():
          tf.import_graph_def(graph_def)
          self.graph.finalize()

        self.sess = tf.Session(graph = self.graph)

      elif ".tflite" in whole_model:
        self.whole_lite = True
        print("load tensorflow lite model")
        experimental_delegates = []
        if "full_quant" in whole_model:
          self.template_image_quant = True
          self.search_image_quant = True
          print("this model is fully quantized")
        if "edgetpu" in whole_model:
          print("this model is for edgetpu")
          experimental_delegates.append(tflite.load_delegate('libedgetpu.so.1'))
        self.interpreter = tflite.Interpreter(model_path=whole_model, experimental_delegates=experimental_delegates)
        self.interpreter.allocate_tensors()
        self.lite_input_details = self.interpreter.get_input_details()
        self.lite_output_details = self.interpreter.get_output_details()
        for input in self.lite_input_details:
          if input['name'] == "input_image":
            self.num_scales = input['shape'][0]

      else:
        raise ValueError("unsupported whole mode")

    else:
      # separate model
      experimental_delegates = []
      if 'edgetpu' in template_model:
        experimental_delegates.append(tflite.load_delegate('libedgetpu.so.1'))
      self.interpreter_template = tflite.Interpreter(template_model, experimental_delegates=experimental_delegates)
      experimental_delegates = []
      if 'edgetpu' in search_model:
        experimental_delegates.append(tflite.load_delegate('libedgetpu.so.1'))
      self.interpreter_search = tflite.Interpreter(search_model, experimental_delegates=experimental_delegates)
      experimental_delegates = []
      if 'edgetpu' in cross_model:
        experimental_delegates.append(tflite.load_delegate('libedgetpu.so.1'))
      self.interpreter_cross = tflite.Interpreter(cross_model, experimental_delegates=experimental_delegates)

      self.interpreter_template.allocate_tensors()
      self.interpreter_search.allocate_tensors()
      self.interpreter_cross.allocate_tensors()
      self.template_input_details = self.interpreter_template.get_input_details()
      self.template_output_details = self.interpreter_template.get_output_details()
      self.search_input_details = self.interpreter_search.get_input_details()
      self.search_output_details = self.interpreter_search.get_output_details()
      self.cross_input_details = self.interpreter_cross.get_input_details()
      self.cross_output_details = self.interpreter_cross.get_output_details()
      self.embed_z_scale = 1.0
      self.embed_z_offset = 0.0
      if 'full_quant' in template_model:
        self.embed_z_scale = self.search_output_details[0]['quantization_parameters']['scales'][0]
        self.embed_z_offset = self.search_output_details[0]['quantization_parameters']['zero_points'][0]
        self.template_image_quant = True

      self.embed_x_scale = 1.0
      self.embed_x_offset = 0.0
      if 'full_quant' in search_model:
        self.embed_x_scale = self.search_output_details[0]['quantization_parameters']['scales'][0]
        self.embed_x_offset = self.search_output_details[0]['quantization_parameters']['zero_points'][0]
        self.search_image_quant = True
      if self.search_input_details[0]['shape'][0] != 1:
        raise ValueError("the search input for seperate model should of one batch for any scale tracking")

      self.cross_input_template_scale = 1.0
      self.cross_input_template_offset = 0.0
      self.cross_input_search_scale = 1.0
      self.cross_input_search_offset = 0.0
      self.cross_output_scale = 1.0
      self.cross_output_offset = 0.0

      self.num_scales = self.cross_input_details[1]['shape'][0]

      self.embed_z = None
      self.update_template = False

    with open(osp.join(config_filepath, 'model_config.json'), 'r') as f:
      self.model_config = json.load(f)
    with open(osp.join(config_filepath, 'track_config.json'), 'r') as f:
      self.track_config = json.load(f)
    self.search_image_size = self.track_config['x_image_size']  # search image size
    self.template_image_size = self.model_config['z_image_size']  # template image size

    self.search_center = np.array([get_center(self.search_image_size),
                                   get_center(self.search_image_size)])

    self.window_influence = self.track_config['window_influence']
    self.current_target_state = None
    self.window = None  # Cosine window
    self.original_target_height = 0
    self.original_target_width = 0

    scales = np.arange(self.num_scales) - get_center(self.num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    self.search_factors = [self.track_config['scale_step'] ** x for x in scales]

  def update_template_image(self, input_image, target_bbox):

    bbox = convert_bbox_format(target_bbox, 'center-based')

    self.original_target_height = bbox.height
    self.original_target_width = bbox.width

    search_images, _ = self.crop_search_image(input_image, bbox)

    # Given the fix ratio btween template image (127) and input image (255) => 1:2
    top = int(round(self.search_center[1] - get_center(self.template_image_size)))
    bottom = int(top + self.template_image_size)
    left = int(round(self.search_center[0] - get_center(self.template_image_size)))
    right = int(left + self.template_image_size)

    template_image = search_images[int(get_center(self.num_scales))][top:bottom, left:right]

    # Update the current_target_state
    self.current_target_state = TargetState(bbox=bbox,
                                            search_pos=self.search_center,
                                            scale_idx=int(get_center(self.num_scales)))

    self.update_template = False

    if self.template_image_quant:
      return template_image.astype(np.uint8)
    else:
      return template_image.astype(np.float32)

  def crop_search_image(self, input_image, target_bbox):
    target_yx = np.array([target_bbox.y, target_bbox.x])
    target_size = np.array([target_bbox.height, target_bbox.width])

    avg_chan = (np.average(input_image, axis=(0, 1))).astype(np.uint8)
    # TODO: to understand the effect of this factor (to much margin from the template image, why_)
    #canonical_size = np.sqrt(np.prod(target_size + 0.5 * np.sum(target_size))) # what is this ?
    canonical_size = np.sqrt(np.prod(target_size + 0.5 * np.sum(target_size))) # what is this ?
    #canonical_size = np.max(target_size) not good for ball, why ?

    search_window_size = self.search_image_size / self.template_image_size * canonical_size
    search_resize_rate = self.search_image_size / search_window_size

    print("search_window_size: {}".format(search_window_size))

    search_images = []
    search_resize_rates = []

    for factor in self.search_factors:
      scaled_search_window_size = factor * search_window_size
      topleft = (target_yx - get_center(scaled_search_window_size))
      bottomright = (target_yx + get_center(scaled_search_window_size))

      search_image = np.ones((int(scaled_search_window_size), int(scaled_search_window_size), 3))
      for i in range(len(avg_chan)):
        search_image[:,:,i] = search_image[:,:,i] * avg_chan[i]

      bottomright = bottomright.astype(np.int32)
      topleft = topleft.astype(np.int32)
      init_x = 0
      init_y = 0
      if topleft[0] < 0: # top
        init_y = -topleft[0]
        topleft[0] = 0
        print ("top violate")
      if topleft[1] < 0: # left
        init_x = -topleft[1]
        topleft[1] = 0
        print ("left violate")
      if bottomright[0] >= input_image.shape[0]: # bottom
        bottomright[0] = input_image.shape[0] - 1
      if bottomright[1] >= input_image.shape[1]: # right
        bottomright[1] = input_image.shape[1] - 1

      # print ("topleft for factor{}: {}".format(factor, topleft))
      # print ("bottomright for factor{}: {}".format(factor, bottomright))

      search_image[init_y: init_y + bottomright[0] - topleft[0], init_x: init_x + bottomright[1] - topleft[1],:] = input_image[topleft[0]:bottomright[0], topleft[1]:bottomright[1],:]

      search_image = cv2.resize(search_image, (self.search_image_size, self.search_image_size))

      if self.search_image_quant:
        search_images.append(search_image.astype(np.uint8))
      else:
        search_images.append(search_image.astype(np.float32))

      search_resize_rates.append(search_resize_rate/factor)

    return search_images, search_resize_rates

  def inference(self, template_image, input_image):

    search_images, search_resize_rates  = self.crop_search_image(input_image, self.current_target_state.bbox)
    #search_images = np.stack([search_images[int(get_center(self.num_scales))] for _ in range(self.num_scales)]) # test

    response = None
    best_scale_index = 0

    if self.separate_mode: # separate
      if not self.update_template:
        dt1 = datetime.datetime.now()
        self.interpreter_template.set_tensor(self.template_input_details[0]['index'], template_image)
        self.interpreter_template.invoke()
        dt2 = datetime.datetime.now()
        print("template inference du: {}".format(dt2.timestamp() - dt1.timestamp()))
        self.embed_z = self.embed_z_scale * (self.interpreter_template.get_tensor(self.template_output_details[0]['index']).astype(np.float32)  - self.embed_z_offset)
        self.update_template = True

      embed_x_list = []
      for i in range(self.num_scales):
        dt1 = datetime.datetime.now()
        self.interpreter_search.set_tensor(self.search_input_details[0]['index'], np.expand_dims(search_images[i],0))
        self.interpreter_search.invoke()
        dt2 = datetime.datetime.now()
        print("search inference du for {}th batch: {}".format(i, dt2.timestamp() - dt1.timestamp()))
        embed_x_list.append(self.embed_x_scale * (self.interpreter_search.get_tensor(self.search_output_details[0]['index']).astype(np.float32)  - self.embed_x_offset))
        #print("embed x : {}".format(embed_x))

      embed_x = np.concatenate(embed_x_list, 0)
      print("embed x : {}".format(embed_x.shape))

      if self.cross_input_template_scale != 1 and self.cross_input_search_scale != 1:
        self.embed_z = (self.embed_z / self.cross_input_template_scale + self.cross_input_template_offset).astype(np.uint8)
        embed_x = (embed_x / self.cross_input_search_scale + self.cross_input_search_offset).astype(np.uint8)

      dt1 = datetime.datetime.now()
      self.interpreter_cross.set_tensor(self.cross_input_details[0]['index'], self.embed_z)
      self.interpreter_cross.set_tensor(self.cross_input_details[1]['index'], embed_x)
      self.interpreter_cross.invoke()
      dt2 = datetime.datetime.now()
      print("cross inference du: {}".format(dt2.timestamp() - dt1.timestamp()))
      raw_output_data = self.cross_output_scale * (self.interpreter_cross.get_tensor(self.cross_output_details[0]['index']) - self.cross_output_offset)
      # print("cross correlation : {}".format(np.squeeze(raw_output_data[int(get_center(self.num_scales))])))

      # post-processing for upsampling the result
      response = np.empty((self.num_scales, self.track_config['upsample_factor'] * raw_output_data.shape[1], self.track_config['upsample_factor'] * raw_output_data.shape[1]))

      for i in range(self.num_scales):
        response[i] = cv2.resize(np.squeeze(raw_output_data, -1)[i], dsize=None, fx=self.track_config['upsample_factor'], fy=self.track_config['upsample_factor'], interpolation=cv2.INTER_CUBIC)

    else:
      if self.whole_lite == True:
        # https://www.tensorflow.org/lite/guide/inference
        self.interpreter.set_tensor(self.lite_input_details[0]['index'], template_image)
        self.interpreter.set_tensor(self.lite_input_details[1]['index'], search_images)

        dt1 = datetime.datetime.now()
        self.interpreter.invoke()
        dt2 = datetime.datetime.now()
        print("inference du: {}".format(dt2.timestamp() - dt1.timestamp()))

        raw_output_data = self.interpreter.get_tensor(self.lite_output_details[0]['index'])

        # print("cross correlation : {}".format(np.squeeze(raw_output_data[int(get_center(self.num_scales))])))

        # post-processing for upsampling the result
        response = np.empty((self.num_scales, self.track_config['upsample_factor'] * raw_output_data.shape[1], self.track_config['upsample_factor'] * raw_output_data.shape[1]))

        for i in range(self.num_scales):
          response[i] = cv2.resize(np.squeeze(raw_output_data, -1)[i], dsize=None, fx=self.track_config['upsample_factor'], fy=self.track_config['upsample_factor'], interpolation=cv2.INTER_CUBIC)

      else:
        output_tensor = self.graph.get_tensor_by_name("import/upsample/final_result:0")
        dt1 = datetime.datetime.now()
        response = self.sess.run(output_tensor, feed_dict = {"import/template_image:0": template_image, "import/input_image:0": search_images})
        dt2 = datetime.datetime.now()
        response = np.squeeze(response, -1)
        #raise ValueError("test")

        print("inference du: {}".format(dt2.timestamp() - dt1.timestamp()))

    # Choose the scale whole response map has the highest peak
    best_scale = 0
    if self.num_scales > 1:
      response_max = np.max(response, axis=(1, 2))
      penalties = self.track_config['scale_penalty'] * np.ones(self.num_scales)
      current_scale_idx = int(get_center(self.num_scales))
      penalties[current_scale_idx] = 1.0
      response_penalized = response_max * penalties
      #print(response_penalized)
      best_scale = np.argmax(response_penalized)

    print(best_scale)
    response = np.squeeze(response[best_scale])

    with np.errstate(all='raise'):  # Raise error if something goes wrong
      response = response - np.min(response)
      response = response / np.sum(response)

    if self.window is None:
      window = np.dot(np.expand_dims(np.hanning(response.shape[1]), 1),
                      np.expand_dims(np.hanning(response.shape[1]), 0))
      self.window = window / np.sum(window)  # normalize window

    response = (1 - self.window_influence) * response + self.window_influence * self.window

    # Find maximum response
    r_max, c_max = np.unravel_index(response.argmax(), response.shape)

    p_coor = np.array([r_max, c_max])
    disp_instance_final = p_coor - get_center(response.shape[1])
    upsample_factor = self.track_config['upsample_factor']
    disp_instance_feat = disp_instance_final / upsample_factor
    # ... Avoid empty position ...
    r_radius = int(response.shape[1] / upsample_factor / 2)
    disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
    # ... in instance input ...
    disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
    # ... in instance original crop (in frame coordinates)
    disp_instance_frame = disp_instance_input / search_resize_rates[best_scale]
    # Position within frame in frame coordinates
    y = self.current_target_state.bbox.y
    x = self.current_target_state.bbox.x
    y += disp_instance_frame[0]
    x += disp_instance_frame[1]

    # Target scale damping and saturation
    target_scale = self.current_target_state.bbox.height / self.original_target_height
    search_factor = self.search_factors[best_scale]
    scale_damp = self.track_config['scale_damp']  # damping factor for scale update
    target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
    target_scale = np.maximum(0.2, np.minimum(5.0, target_scale)) # heuristic

    # Some book keeping
    height = self.original_target_height * target_scale
    width = self.original_target_width * target_scale
    self.current_target_state.bbox = Rectangle(x, y, width, height)
    self.current_target_state.scale_idx = best_scale
    self.current_target_state.search_pos = self.search_center + disp_instance_input

    outputs = {'search_image': search_images[best_scale_index], 'response': response, 'current_target_state': self.current_target_state}
    return outputs

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--models_dir', dest='models_dir', action="store",
                      default='Logs/SiamFC/track_model_checkpoints/train/models', type=str)
  parser.add_argument('--config', dest='config_filepath', action="store",
                      help='the path of tracking config for inference', default='Logs/SiamFC/track_model_checkpoints/train', type=str)

  parser.add_argument('--images', dest='image_filepath', action="store",
                      help='the path of iamges to do inference', default='assets/drone', type=str)

  parser.add_argument('--headless', dest='headless', action="store_true")

  parser.add_argument('--whole_model', dest='whole_model', action="store",
                      help='the path of inference model for whole sequence', default='whole_model_scale1.pb', type=str)

  parser.add_argument('--search_model', dest='search_model', action="store",
                      default='search_image_feature_extractor_full_quant_scale1_edgetpu.tflite', type=str)
  parser.add_argument('--template_model', dest='template_model', action="store",
                      default='template_image_feature_extractor_scale1.tflite', type=str)
  parser.add_argument('--cross_model', dest='cross_model', action="store",
                      default='cross_correlation_scale1.tflite', type=str)

  parser.add_argument('--separate_mode', dest='separate_mode', action="store_true")

  args, _ = parser.parse_known_args()

  first_line = open(args.image_filepath + '/groundtruth_rect.txt').readline()
  bbox = [int(v) for v in first_line.strip().split(',')]
  init_bbox = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])  # 0-index in python

  tracker = SiameseTracking(args.config_filepath,
                            args.separate_mode,
                            osp.join(args.models_dir, args.whole_model),
                            osp.join(args.models_dir, args.template_model),
                            osp.join(args.models_dir, args.search_model),
                            osp.join(args.models_dir, args.cross_model))

  filenames = sort_nicely(glob(args.image_filepath + '/img/*.jpg'))

  first_image = cv2.imread(filenames[0])
  template_image = tracker.update_template_image(first_image, init_bbox)

  if not args.headless:
    cv2.imshow('template_image',template_image.astype(np.uint8))

  for i, filename in enumerate(filenames):
    if i > 0:
      input_image = cv2.imread(filenames[i])
      dt1 = datetime.datetime.now()
      outputs = tracker.inference(template_image, input_image)
      dt2 = datetime.datetime.now()
      print("tracking du: {}".format(dt2.timestamp() - dt1.timestamp()))

      # visualize
      search_image = outputs['search_image'].astype(np.uint8)

      bbox_search = convert_bbox_format(outputs['current_target_state'].bbox, 'top-left-based')
      input_image = cv2.rectangle(input_image,(int(bbox_search.x), int(bbox_search.y)),(int(bbox_search.x+bbox_search.width), int(bbox_search.y+bbox_search.height)),(0,255,0),2)

      if not args.headless:
        cv2.imshow('search_image',search_image)
        cv2.imshow('raw_image', input_image)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
          sys.exit()

