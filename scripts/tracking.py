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
    def __init__(self, model_filepath, config_filepath):

        self.lite = False
        self.full_quant = False
        self.edgetpu = False
        self.num_scales = 1

        # check the model file
        if ".pb" in args.model_filepath:
          print("load frozen graph model")
        elif ".tflite" in args.model_filepath:
          self.lite = True
          print("load tensorflow lite model")
          if "full_quant" in args.model_filepath:
            self.full_quant = True
            print("this model is fully quantized")
          if "edgetpu" in args.model_filepath:
            self.edgetpu = True
            print("this model is for edgetpu")
        else:
          raise ValueError("test")

        if self.lite == True:
          if self.edgetpu:
            self.interpreter = tflite.Interpreter(model_path=model_filepath, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
          else:
            self.interpreter = tflite.Interpreter(model_path=model_filepath)

          self.interpreter.allocate_tensors()
          self.lite_input_details = self.interpreter.get_input_details()
          self.lite_output_details = self.interpreter.get_output_details()
          for input in self.lite_input_details:
            if input['name'] == "input_image":
              self.num_scales = input['shape'][0]
          print("=========== the scale number of model: {}".format(self.num_scales))
        else:
          print('Loading frozen graphmodel...')
          self.graph = tf.Graph()

          with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            for n in graph_def.node:
              if n.op in ('Placeholder'):
                if n.name == "input_image":
                  self.num_scales = n.attr['shape'].shape.dim[0].size

            print("=========== the scale number of model: {}".format(self.num_scales))

          with self.graph.as_default():
            tf.import_graph_def(graph_def)
            self.graph.finalize()

            self.sess = tf.Session(graph = self.graph)

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
      print(len(search_images))
      # Given the fix ratio btween template image (127) and input image (255) => 1:2
      top = int(round(self.search_center[1] - get_center(self.template_image_size)))
      bottom = int(top + self.template_image_size)
      left = int(round(self.search_center[0] - get_center(self.template_image_size)))
      right = int(left + self.template_image_size)

      template_image = search_images[int(get_center(self.num_scales))][top:bottom, left:right]
      print("template_image: {}".format(template_image.shape))

      # Update the current_target_state
      self.current_target_state = TargetState(bbox=bbox,
                                              search_pos=self.search_center,
                                              scale_idx=int(get_center(self.num_scales)))

      if self.full_quant == True:
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

        if self.full_quant == True:
          search_images.append(search_image.astype(np.uint8))
        else:
          search_images.append(search_image.astype(np.float32))

        search_resize_rates.append(search_resize_rate/factor)

      return search_images, search_resize_rates

    def inference(self, template_image, input_image):

      search_images, search_resize_rates  = self.crop_search_image(input_image, self.current_target_state.bbox)
      # search_image = np.stack([search_image for _ in range(self.num_scales)]) # test

      response = None
      best_scale_index = 0
      if self.lite == True:
        # https://www.tensorflow.org/lite/guide/inference
        self.interpreter.set_tensor(self.lite_input_details[0]['index'], template_image)
        self.interpreter.set_tensor(self.lite_input_details[1]['index'], search_images)

        dt1 = datetime.datetime.now()
        self.interpreter.invoke()
        dt2 = datetime.datetime.now()
        print("inference du: {}".format(dt2.timestamp() - dt1.timestamp()))

        raw_output_data = self.interpreter.get_tensor(self.lite_output_details[0]['index'])

        #raise ValueError("test")

        # post-processing for upsampling the result
        response = np.empty((self.num_scales, self.track_config['upsample_factor'] * raw_output_data.shape[1], self.track_config['upsample_factor'] * raw_output_data.shape[1]))

        for i in range(self.num_scales):
          response[i] = cv2.resize(np.squeeze(raw_output_data, -1)[i], dsize=None, fx=self.track_config['upsample_factor'], fy=self.track_config['upsample_factor'], interpolation=cv2.INTER_CUBIC)

      else:
        output_tensor = self.graph.get_tensor_by_name("import/upsample/final_result:0")
        dt1 = datetime.datetime.now()
        response = self.sess.run(output_tensor, feed_dict = {"import/template_image:0": template_image, "import/input_image:0": search_images})
        dt2 = datetime.datetime.now()
        #print(np.squeeze(response))
        #print(response.shape)
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
  parser.add_argument('--model', dest='model_filepath', action="store",
                      help='the path of frozen graph model for inference', default='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained/models/whole_model_scale1.pb', type=str)
  parser.add_argument('--config', dest='config_filepath', action="store",
                      help='the path of tracking config for inference', default='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained', type=str)

  parser.add_argument('--images', dest='image_filepath', action="store",
                      help='the path of iamges to do inference', default='assets/drone', type=str)

  parser.add_argument('--headless', dest='headless', action="store_true")

  args, _ = parser.parse_known_args()

  first_line = open(args.image_filepath + '/groundtruth_rect.txt').readline()
  bbox = [int(v) for v in first_line.strip().split(',')]
  init_bbox = Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])  # 0-index in python

  tracker = SiameseTracking(args.model_filepath, args.config_filepath)

  filenames = sort_nicely(glob(args.image_filepath + '/img/*.jpg'))

  first_image = cv2.imread(filenames[0])
  template_image = tracker.update_template_image(first_image, init_bbox)

  if not args.headless:
    cv2.imshow('template_image',template_image.astype(np.uint8))
    #first_image = cv2.rectangle(first_image,(int(init_bbox.x), int(init_bbox.y)),(int(init_bbox.x+init_bbox.width), int(init_bbox.y+init_bbox.height)),(0,255,0),2)
    #cv2.imshow('first_image', first_image.astype(np.uint8))


  for i, filename in enumerate(filenames):
    if i >= 0:
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

