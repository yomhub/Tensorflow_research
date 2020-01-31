import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
from mydataset.svt import SVT
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.tflib.evaluate_tools import draw_boxes, save_image
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
  # rr = np.array([[0,1,1],[2,3,3]])
  mydatalog = SVT()
  for i in range(4):
    x_train, y_train = mydatalog.read_train_batch(70)
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
  