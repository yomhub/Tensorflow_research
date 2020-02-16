import os, sys
import tensorflow as tf
import numpy as np
import argparse
# from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.config import cfg
from lib.model.label_rcnn import Label_RCNN, LRCNNLoss
from lib.frcnn_trainer import FRCNNTrainer
from lib.tflib.bbox_transform import *
# 

if __name__ == "__main__":
  mydatalog = SVT(out_size=[1021,1025])
  x_train, y_train = mydatalog.read_train_batch(1)
  y_train = xywh2yxyx(y_train[0])
  tt2 = gen_label_with_width_from_gt((33,32),y_train,[1025,1021])
  # loss = LRCNNLoss((1025,1021),gtformat='xywh')
  # tfz = tf.zeros((1,1025,1021,3))
  # model = Label_RCNN()
  # model(tfz)
  # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  # for i in range(20):
  #   x_train, y_train = mydatalog.read_train_batch(5)
  #   for 
  #   with tf.GradientTape(persistent=False) as tape:
  #     tape.watch(model.trainable_variables)
  #     pred = model(x_train/256.0)
  #     loss = loss(y_train[0],pred)
  #   grads = tape.gradient(loss, model.trainable_variables)
  #   opt.apply_gradients(zip(grads, model.trainable_variables))
  # mydatalog = SVT(out_size=[512,512])
  # pred = model(x_train)
  # yy_true = xywh2yxyx(y_train[0][:,1:])
  # loss_value = pre_box_loss(yy_true,pred["l1_bbox"],[512,512])
  # path_list, mask_np = build_boxex_from_path(pred["l1_score"],pred["l1_bbox"],pred["l1_ort"],1)
  # yy_true = map2coordinate(yy_true,[512,512],pred["l1_score"].shape[1:3])
  # label_list = get_label_from_mask(yy_true,mask_np)
  # pred = loss(y_train[0],pred)
  print()