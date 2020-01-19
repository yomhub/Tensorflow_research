import os, sys
# import tensorflow as tf
# import numpy as np
# from mydataset.ctw import CTW
# from lib.tflib.evaluate_tools import *
# import matplotlib.pyplot as plt
import argparse
# mydatalog = CTW(out_size=[512,512])

if __name__ == "__main__":

# x_train, y_train = mydatalog.read_batch(batch_size=1)

# y_train = tf.stack(
#   [y_train[:,2],y_train[:,1],y_train[:,2]+y_train[:,4],y_train[:,1]+y_train[:,3]],
#   axis=1)
# img = draw_boxes(x_train,y_train)
# show_img(img)

  parser = argparse.ArgumentParser(description='Choose settings.')
  parser.add_argument('--proposal', help='Choose proposal in nms and top_k.',default='nms')
  parser.add_argument('--debug', help='Set --debug if want to debug.', action="store_true")
  parser.add_argument('--step', type=int, help='Step size.',default=50)
  parser.add_argument('--batch', type=int, help='Batch size.',default=20)
  parser.add_argument('--learnrate', type=float, help='Batch size.',default=0.001)
  args = parser.parse_args()
  print(args.learnrate)
  print(type(args.learnrate))
