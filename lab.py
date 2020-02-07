import os, sys
import tensorflow as tf
import numpy as np
import argparse
# from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.config import cfg
from lib.model.label_rcnn import Label_RCNN
from lib.frcnn_trainer import FRCNNTrainer
# 

if __name__ == "__main__":
  # mydatalog = SVT(out_size=[args.datax,args.datay])
  tfz = tf.zeros((1,512,512,3))
  model = Label_RCNN()
  _=model(tfz)
  print(len(_))