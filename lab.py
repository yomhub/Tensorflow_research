import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
# from mydataset.svt import SVT
# from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
# from lib.tflib.evaluate_tools import draw_boxes, save_image
# from lib.tflib.log_tools import _auto_histogram
# import matplotlib.pyplot as plt
import argparse
# 

if __name__ == "__main__":
# y_train = tf.stack(
#   [y_train[:,2],y_train[:,1],y_train[:,2]+y_train[:,4],y_train[:,1]+y_train[:,3]],
#   axis=1)
# img = draw_boxes(x_train,y_train)
# show_img(img)

  # model = Faster_RCNN(num_classes=2)
  # mydatalog = CTW(out_size=[512,512])
  # mydatalog = CTW()
  # x_train, y_train = mydatalog.read_batch(batch_size=10)
  # for i in range(len(y_train)):
  #   img = draw_boxes(x_train[i],y_train[i][:,1:])
  #   save_image(img,"{}.jpg".format(i))
  tt = tf.random.uniform([5,4])
  zz = tf.zeros(tt.shape)

  log_m = tf.math.logical_and((tt[:,2]>0 & tt[:,0]>0),tt[:,2]<110)
  log_m = tf.math.logical_and(log_m,tt[:,1]>0)
  
  log_m = tf.tile(tf.reshape(log_m,[log_m.shape[0],1]),[1,4])
  tt = tf.where(log_m,tt,zz)
  # mydatalog = SVT()
  # tt = mydatalog.caculate_avg()
  # with tf.summary.create_file_writer("logss") as we:
  #   we.as_default()
  #   _auto_histogram(tt,logname="svt")
  # tf.summary.histogram(logname,dic_data,step=step)
  # for i in range(4):
  #   x_train, y_train = mydatalog.read_train_batch(70)
  # vgg16=tf.keras.applications.VGG16(weights='imagenet', include_top=False)
  # feature_model_t = tf.keras.Model(
  #       inputs=vgg16.input,
  #       outputs=[
  #         vgg16.get_layer('block3_pool').output,
  #         vgg16.get_layer('block4_conv3').output,
  #         vgg16.get_layer('block4_pool').output,
  #         vgg16.get_layer('block5_conv3').output
  #       ],
  #       # name=""
  #     )
  # front_feature = feature_model_t(tf.zeros((1,512,512,3),dtype=tf.float32))
  # # model(tf.zeros((1,512,512,3),dtype=tf.float32))
  # ll = {
  #   "t1":1,
  #   "t2":2,
  # }
  # for itm in ll:
  #   print(itm)
  
  print()
  