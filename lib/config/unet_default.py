from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import tensorflow as tf

# from config.unet_default import cfg
cfg = {}

# initial learning rate
cfg['LEARNING_RATE'] = 0.001

# choose feature network
['vgg16', 'resnet']
cfg['F_NET'] = 'vgg16'
# choose whether to use pre trained network
cfg['F_NET_PT'] = 'imagenet'
# choose feature layers
# ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
cfg['F_LAYER'] = {
  'vgg16': [
    # 'block1_pool', # 1/2, ch64 RF6*6 PIk2, after 3x3conv rf = 10
    # 'block2_conv2', # ch128 RF14*14 PIk2
    # 'block2_pool', # 1/4, ch128 RF16*16 PIk4, after 3x3conv rf = 24, fls = 8, flstr = 2
    'block3_conv3', # ch256 RF40*40 PIk4
    # 'block3_pool', # 1/8, ch256 RF44*44 PIk8
    'block4_conv3', # ch512 RF92*92 PIk8
    # 'block4_pool', # 1/16, ch512 RF100*100 PIk16 
    'block5_conv3', # ch512 RF196*196 PIk16
    # 'block5_pool', # 1/32, ch512 RF212*212 PIk32
    # 'fc6',
    'fc7',
    ],
  'resnet': [
    'pool1_pool', # 1/4, ch=64
    'conv2_block3_out', # 1/4, ch=256
    'conv3_block4_out', # 1/8, ch=512
    'conv4_block6_out', # 1/16, ch=1024
    # 'conv5_block3_out', # 1/32, ch=2048
    ],
}
# max scale of feature image sieze
cfg['F_MXS'] = {
  'vgg16': 32,
  'resnet': 16,
}
# set final fused channels 
cfg['F_FCHS'] = {
  'vgg16': 64,
  'resnet': 64,
}

# score map activation
cfg['SRC_ACT'] = None
# score map channels num
cfg['SRC_CHS'] = 2


# negative / postive sample ratio
cfg['NP_RATE'] = 2/1

# pixel class loss weight
cfg['PX_CLS_LOSS_W'] = 0.0
# maxium negative pixel num
cfg['PX_CLS_LOSS_MAX_NEG'] = 10000