import os, sys
import tensorflow as tf
import numpy as np
from lib.model.config import cfg

if __name__ == "__main__":
  
  test = tf.Variable([[0,1,2,3],[-4,-5,-6,-7],[0,1,0,1],[0,1,0,2],[0,1,0,3],[0,1,0,4]],dtype=tf.int64)
  test2 = tf.Variable([[0,1,2,3]],dtype=tf.int64)
  inds=tf.reshape(tf.range(50,dtype=tf.int64),[1,-1])

  print(inds)
