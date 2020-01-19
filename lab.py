import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
# from lib.tflib.evaluate_tools import *
import matplotlib.pyplot as plt

# mydatalog = CTW(out_size=[512,512])

# x_train, y_train = mydatalog.read_batch(batch_size=1)

# y_train = tf.stack(
#   [y_train[:,2],y_train[:,1],y_train[:,2]+y_train[:,4],y_train[:,1]+y_train[:,3]],
#   axis=1)
# img = draw_boxes(x_train,y_train)
# show_img(img)

ff = open("tt.txt",'a+',encoding='utf8')
ff.write("l;ine1\n")
ff.close()
ff = open("tt.txt",'a+',encoding='utf8')
ff.write("l;ine1\n")
ff.close()