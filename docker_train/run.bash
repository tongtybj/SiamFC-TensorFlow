#!/usr/bin/env bash

#--userns=host: https://dockerlabs.collabnix.com/advanced/security/userns/

docker run --name siamese-fc-train \
       --gpus all \
       --rm \
       --mount type=bind,src=$HOME/SiamFC-TensorFlow,dst=/root/SiamFC-TensorFlow \
       --mount type=bind,src=${1:-~/ILSVRC2015},dst=/root/ILSVRC2015 \
       -it --privileged -p 6006:6006 \
       --userns=host \
       siamese-fc-train
