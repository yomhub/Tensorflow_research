# Base data loader class
import os,sys
import numpy as np
import tensorflow as tf

class LoaderBase():
  def __init__(self, out_size=None, out_format='list'):
    self.out_y = out_size[0]
    self.out_x = out_size[1]
    self.out_size = out_size
    self.out_format = 'tensor' if(out_format.lower()=='tensor') else 'list'
    self.train_conter = 0
    self.test_conter = 0
  
  def read_train_batch(self, batch_size=10):
    raise NotImplementedError

  def read_test_batch(self, batch_size=10):
    raise NotImplementedError

  def set_conter(self,train_conter=None,test_counter=None):
    if(train_conter):
      self.train_conter = train_conter
    if(test_counter):
      self.test_conter = test_counter

  def read_single_image(self,dir):
    return tf.image.resize(tf.image.decode_image(tf.io.read_file(dir)),self.out_size)

  def find_slice(self,counter,total,batch_size):
    """
      Read helper
      Args:
        counter: current counter
        total: total data num
        batch_size: read batch size
      Return:
        counter: counter after reading
        slice_a: slice
        slice_b: None or slice
      Usage:
        list[slice_a] + list[slice_b] if(slice_b)else list[slice_a]
    """
    mend = min(total,counter+batch_size)
    slice_a = slice(counter,mend) 
    readnum = mend-counter
    counter += readnum
    if(readnum < batch_size):
      slice_b = slice(batch_size - readnum)
      counter = batch_size - readnum
    else:
      slice_b = None

    return counter,slice_a,slice_b