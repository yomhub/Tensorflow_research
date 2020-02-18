#!/bin/bash
conda init bash
conda activate tf

loaddir = ""
if [ ! -n "$1" ] ;then
    loaddir = "log/"$1
else
    loaddir = "log/"
fi
# echo $loaddir
tensorboard --logdir=log/LRCNN_tv_set_with_svt_adam --host=0.0.0.0