#!/usr/bin/env bash

#--userns=host: https://dockerlabs.collabnix.com/advanced/security/userns/

docker run --name siamese-fc-inference \
       --gpus all \
       --rm \
       --mount type=bind,src=$HOME/SiamFC-TensorFlow,dst=/root/SiamFC-TensorFlow \
       -it --privileged -p 6006:6006 \
       --userns=host \
       siamese-fc-inference
