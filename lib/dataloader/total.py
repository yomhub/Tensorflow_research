# Total text dataset
import os
import sys
import tarfile
import math
import numpy as np
import tensorflow as tf

_LOCAL_DIR = os.path.split(__file__)[0]
_ANN_FILENAME = 'ctw-annotations.tar.gz'
_STW_DIR = os.path.join(_LOCAL_DIR, 'ctw')
_TRAIN_DIR = os.path.join(_LOCAL_DIR, 'ctw', 'train')
_TEST_DIR = os.path.join(_LOCAL_DIR, 'ctw', 'test')
_LOG_FILE = os.path.join(_LOCAL_DIR, 'ctw', 'log.txt')
_TOTAL_TRAIN_NUM = 1000 * 25 + 887
_TOTAL_TEST_NUM = 1000 * 6 + 398

class TTText():
  """
    Args:

      out_size: [h,w]
      gt_format: string, mask or gtbox
      out_format: string, list or tensor
  """
  def __init__(self, dir, out_size=[720,1280], gt_format='mask', out_format='list'):
    gt_format = gt_format.lower()
    if(gt_format=='mask'):
      self.gt_format = gt_format
    else:
      self.gt_format = 'gtbox'
    
    self.xtraindir = os.path.join(dir,'Images','Train')
    self.xtestdir = os.path.join(dir,'Images','Test')
    self.ytraindir = os.path.join(dir,'gt_pixel','Train')
    self.ytestdir = os.path.join(dir,'gt_pixel','Test')

    self.out_y = out_size[0]
    self.out_x = out_size[1]
    self.out_size = out_size
    self.out_format = 'tensor' if(out_format.lower()=='tensor') else 'list'
    self.train_conter = 0
    self.test_conter = 0
    self.train_img_names = None
    self.test_img_names = None
    for root, dirs, files in os.walk(self.xtraindir):
      self.train_img_names = [name for name in files if (os.path.splitext(name)[-1] == ".jpg" or
        os.path.splitext(name)[-1] == ".png" or
        os.path.splitext(name)[-1] == ".bmp")]
    for root, dirs, files in os.walk(self.xtestdir):
      self.test_img_names = [name for name in files if (os.path.splitext(name)[-1] == ".jpg" or
        os.path.splitext(name)[-1] == ".png" or
        os.path.splitext(name)[-1] == ".bmp")]

    self.total_train = len(self.train_img_names)
    self.total_test = len(self.test_img_names)

  def read_train_batch(self, batch_size=10):
    img_names = self.train_img_names
    cur_conter, slice_a, slice_b = self.find_slice(self.train_conter,self.total_train,batch_size)
    img_list = []
    msk_list = []
    dirs = (img_names[slice_a] + img_names[slice_b]) if(slice_b)else img_names[slice_a]

    for mdir in dirs:
      tmp = tf.image.resize(
        tf.image.decode_image(tf.io.read_file(os.path.join(self.xtraindir,mdir))),self.out_size)
      img_list.append(tf.reshape(tmp,[1]+tmp.shape))
      msk_list.append(tf.image.resize(
        tf.image.decode_image(tf.io.read_file(os.path.join(self.ytraindir,mdir))),self.out_size,'nearest'))

    if(self.out_format=='tensor'):
      img_list = tf.convert_to_tensor(img_list)
      if(self.gt_format=='mask'):
        msk_list = tf.convert_to_tensor(msk_list)

    self.train_conter = cur_conter
    return img_list, msk_list

  def read_test_batch(self, batch_size=10):
    img_names = self.test_img_names
    cur_conter, slice_a, slice_b = self.find_slice(self.test_conter,self.total_test,batch_size)
    img_list = []
    msk_list = []
    dirs = (img_names[slice_a] + img_names[slice_b]) if(slice_b)else img_names[slice_a]

    for mdir in dirs:
      tmp = tf.image.resize(
        tf.image.decode_image(tf.io.read_file(os.path.join(self.xtestdir,mdir))),self.out_size)
      img_list.append(tf.reshape(tmp,[1]+tmp.shape))
      msk_list.append(tf.image.resize(
        tf.image.decode_image(tf.io.read_file(os.path.join(self.ytestdir,mdir))),self.out_size,'nearest'))

    if(self.out_format=='tensor'):
      img_list = tf.convert_to_tensor(img_list)
      if(self.gt_format=='mask'):
        msk_list = tf.convert_to_tensor(msk_list)
        
    self.test_conter = cur_conter
    return img_list, msk_list

  def set_conter(self,train_conter=None,test_counter=None):
    if(train_conter):
      self.train_conter = train_conter
    if(test_counter):
      self.test_conter = test_counter
    
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

