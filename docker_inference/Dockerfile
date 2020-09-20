# docker build -t ubuntu1604py35
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y build-essential python3.5 python3.5-dev python3-pip python3.5-venv
RUN apt-get install -y git emacs  curl

# update pip
RUN python3.5 -m pip install pip --upgrade
RUN python3.5 -m pip install wheel
RUN pip install -U setuptools

# sacred
RUN pip install sacred==0.7.5
RUN pip install pymongo

# pillow for image processing
RUN pip install pillow

# opencv 
RUN pip install opencv-python
RUN apt-get install -y libgl1-mesa-dev # for libGL.so

# scipy
RUN pip install scipy

# tensorflow-gpu (2.1.0)
RUN pip install tensorflow-gpu==2.1.0

WORKDIR /root

# nets for mobilenent1
RUN git clone https://github.com/tensorflow/models.git
ENV PYTHONPATH $PYTHONPATH:/root/models/research/slim

# also necesary for moblinenet_v1
RUN git clone https://github.com/google-research/tf-slim.git
ENV PYTHONPATH $PYTHONPATH:/root/tf-slim

# tflite-runtime: version 2.1.0 for python3.5
RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl

# edgetpu-compiler to convert edgetpu compatible tflite model
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get install -y edgetpu-compiler

# edgetpulib to inference in edgetpu device
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

RUN apt-get install -y libedgetpu1-std

RUN mkdir -p SiamFC-TensorFlow/
WORKDIR SiamFC-TensorFlow

RUN echo alias python="python3" >> ~/.bashrc

