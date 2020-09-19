# SiamFC-TensorFlow
A TensorFlow implementation of the SiamFC tracker

## Installation
-  [tensorflow/models/resaerch/slim](https://github.com/google-research/tf-slim) with `pip install -e .` (setup.py), and install [tf-slim](https://github.com/google-research/tf-slim) with `pip install -e .` (setup.py)

- [tflite_runtime](https://www.tensorflow.org/lite/guide/python)(tflite_runtime-2.1.0: compatible for ubuntu16.04, python3.5 + tf2.1.0) instead of `tensorflow.lite`, this is also necessary for delagatation of `libedgetpu.so.1` (tflite-2.1.0 is OK for edgeput)

- [edgetpu library](https://coral.ai/docs/accelerator/get-started/#1b-on-mac)


## Training
```bash
# 1. Download and unzip the ImageNet VID 2015 dataset (~86GB)
# Now, we assume it is unzipped in /path/to/ILSVRC2015

DATASET=/path/to/ILSVRC2015
cd SiamFC-TensorFlow
ln -s $DATASET data/ILSVRC2015

# 2. Prepare training data
# If you have followed the data preparation procedures in 
# the MatConvNet implementation, simply create a soft link 
# pointing to the curated dataset:
#       ln -s $CURATED_DATASET data/ILSVRC2015-VID-Curation
# Otherwise, create it from scratch by

python scripts/preprocess_VID_data.py

# 3. Split train/val dataset and store corresponding image paths

python scripts/build_VID2015_imdb.py

# 3. Start training
# please add options in train.py based on sacred rules
# You can get quite good results after ~70k iterations.
python experiments/train.py

# 4. View the training progress in TensorBoard
# Open a new terminal session and cd to SiamFC-TensorFlow, then
tensorboard --logdir=Logs/SiamFC/track_model_checkpoints/train/
```

## Export Inference  Model

### Export whole model frozen graph
```
$ python scripts/export/export_whole_model_frozen_graph.py
```
**note**: `--checkpoint_dir=Logs/SiamFC/track_model_checkpoints/train` for designating the training directory.

### Export common tflite model and a fully quantized tflite model
```
$ python scripts/export/export_whole_model_tflite_model.py
```
**note1**: please use `--frozen_graph_model` to designate a proper *.pb file.
**note2**: need tf-2.1.0, since version < 2.1.0 does no support the quantization of SPLIT
```
  $ pip install pip install tensorflow-gpu==2.1.0
  $ python scripts/export_siamese_fc_model.py
```

**depreacated**: you have to also install [tf-slim](converted_model_full_quant.tflite)

### Export seperate model
```
$ python scripts/export/export_separate_model_tflite_mode.py
```
**note1**: this is based on a frozen graph, option for alexnet: `--frozen_graph_model=/home/chou/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models/whole_model_scale1.pb --config_filepath=/home/chou/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models`.

**note2**: compile to edgetpu compatible tflite mode: `edgetpu_compiler search_image_feature_extractor_full_quant_scale1.tflite`


## Tracking

### From frozen graph
```
$ python scripts/tracking.py
```
**note1**: options: `--models_dir=Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models --whole_mode=whole_model_scale3.pb --config=Logs/SiamFC/track_model_checkpoints/train-alexnet-split`
**note2**:  ~0.02s in GTX1080Ti with  **scale 1?**, ~0.04 in RTX2060 SUPER with scale 1
  
### From tflite model (~0.09s in desktop CPU)

#### tracking using non-quantization model (**~0.?s** in desktop CPU)
```
$ python scripts/tracking.py
```
**note1**: options: `--models_dir=Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models --whole_mode=whole_model_scale3.tflite --config=Logs/SiamFC/track_model_checkpoints/train-alexnet-split`
**note2**: ~0.1 in Intel i7-4770 with scale1 ; ~0.3 in Intel i7-4770 with scale3


#### tracking using full quantization model (different from weight-only quatization model)
```
$ python scripts/tracking.py
```
**note1**: options: `--models_dir=Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models --whole_mode=whole_model_full_quant_scale1.tflite --config=Logs/SiamFC/track_model_checkpoints/train-alexnet-split`
**note2**: ~7.2 in Intel i7-4770 with scale 1

### Using seperate model (non-scale tracking):
```
$ python scripts/tracking.py  --config=/home/chou/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/train-alexnet-split  --models_dir=/home/chou/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/train-alexnet-split/models --search_model=search_image_feature_extractor_scale1.tflite --template_model=template_image_feature_extractor_scale1.tflite --cross_model=cross_correlation_scale1.tflite --headless --separate_mode
```
  ** note1 **: fastest option: `--search_mode=search_image_feature_extractor_full_quant_scale1_edgetpu.tflite`
  ** note2 for scale **: scale1 => `cross_correlation_scale1.tflite`; scale3 => `cross_correlation_scale3.tflite`; do not change `--search_model` to `scale3`.
```
tracking du: 0.0323488712310791
search_window_size: 124.62415099356501
search inference du for 0th batch: 0.004250049591064453
search inference du for 1th batch: 0.004121065139770508
search inference du for 2th batch: 0.004178047180175781
embed x : (3, 22, 22, 256)
cross inference du: 0.002541780471801758
```

## Tips:
1. tensorflow lite quantization:
   tensorflow with version < 2.1.0 does no support the quantization of SPLIT, so the split operation of alexnet can not convert withotu 2.1.0

2. edgetpu_compiler: convert to edgeput compatible .tflite model. For download, please check https://coral.ai/docs/edgetpu/compiler/#usage .
edgetpu compiler is only compatible with tensorflow 1.15.0, the .tflite model converted by tensorflow 2.x can not compiler to edgetpu model: https://github.com/tensorflow/tensorflow/issues/31368.

6. for mobilnet, please use tf 1.x (e.g. 1.5.0) also the frozen graph export, but for tflite conversion, please use tf 2.x (2.1.0)

7. training: batch size 8-> 64, learning rate 0.01 -> 0.05, context size: 0.5 -> 0.25 ?
