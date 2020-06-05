# Total text dataset
import os
import sys
import tarfile
import math
import numpy as np
import tensorflow as tf

_TOTAL_TRAIN_NUM = 1000 * 25 + 887
_TOTAL_TEST_NUM = 1000 * 6 + 398

def txt_helper(fname,dtype=int,img_resize_coe=None):
  """
    Return boxs [[y1,x1,y2,x2]...] information.
    The text format is:
      One line represent one box, e.g.
      x: [[206 251 386 542 620 646 550 358 189 140]], y: [[ 633  811  931  946  926  976 1009  989  845  629]], ornt: [u'c'], transcriptions: [u'PETROSAINS']
      x = X-coordinate
      y = Y-coordinate
      ornt = Orientation (c=curve; h=horizontal; m=multi-oriented; #=dont care)
      transcriptions  = Text
    @param img_resize_coe: coefficient in [h,w]
  """
  dtype = dtype if(type(dtype)==type)else int
  dt = lambda x:dtype(x)
  boxs = []
  with open(fname,'r') as f:
    tmp = f.readlines()
    tmp = [o.split(', ') for o in tmp]
    try:
      for l in tmp: 
        xs = [dt(o) for o in l[0].split('[[')[-1].split(']]')[0].split()]
        ys = [dt(o) for o in l[1].split('[[')[-1].split(']]')[0].split()]
        if(img_resize_coe!=None):
          boxs.append([dt(1), min(ys)*img_resize_coe[0], min(xs)*img_resize_coe[1],
            max(ys)*img_resize_coe[0], max(xs)*img_resize_coe[1]])
        else:
          boxs.append([dt(1), min(ys), min(xs),max(ys), max(xs)])
    except:
      raise RuntimeError('Read err at {} \nin {}\nin {}'.format(fname,tmp,l))
  return tf.convert_to_tensor(boxs)

class TTText():
  """
    Args:
      out_size: [h,w]
      gt_format: string, mask or gtbox
      out_format: string, list or tensor
      nor: normalization coordinate to [0,1]

  """
  def __init__(self, dir, out_size=[720,1280], gt_format='mask', out_format='list', max_size=2048, nor=True):
    gt_format = gt_format.lower()
    if(gt_format=='mask'):
      self.gt_format = gt_format
    else:
      self.gt_format = 'gtbox'
    
    self.xtraindir = os.path.join(dir,'Images','Train')
    self.xtestdir = os.path.join(dir,'Images','Test')
    self.pixeltraindir = os.path.join(dir,'gt_pixel','Train')
    self.pixeltestdir = os.path.join(dir,'gt_pixel','Test')
    # txt name: 'poly_gt_' + image name + '.txt'
    self.txttraindir = os.path.join(dir,'gt_txt','Train')
    self.txttestdir = os.path.join(dir,'gt_txt','Test')

    self.out_size = out_size
    self.out_format = 'tensor' if(out_format.lower()=='tensor') else 'list'
    self.train_conter = 0
    self.test_conter = 0
    self.train_img_names = None
    self.test_img_names = None
    self.max_size = max_size if(type(max_size)==list)else [max_size,max_size]
    self.train_img_names=[]
    for root, dirs, files in os.walk(self.xtraindir):
      self.train_img_names += [name for name in files if (os.path.splitext(name)[-1] == ".jpg" or
        os.path.splitext(name)[-1] == ".png" or
        os.path.splitext(name)[-1] == ".bmp")]
    self.test_img_names=[]
    for root, dirs, files in os.walk(self.xtestdir):
      self.test_img_names += [name for name in files if (os.path.splitext(name)[-1] == ".jpg" or
        os.path.splitext(name)[-1] == ".png" or
        os.path.splitext(name)[-1] == ".bmp")]
    self.nor = bool(nor)
    self.total_train = len(self.train_img_names)
    self.total_test = len(self.test_img_names)

  def read_train_batch(self, batch_size=10):
    img_names = self.train_img_names
    cur_conter, slice_a, slice_b = self.find_slice(self.train_conter,self.total_train,batch_size)
    img_list = []
    y_list = []
    boxs = []
    dirs = (img_names[slice_a] + img_names[slice_b]) if(slice_b)else img_names[slice_a]
    for mdir in dirs:
      # read image
      tmp = tf.image.decode_image(tf.io.read_file(os.path.join(self.xtraindir,mdir)))
      coe = [1/tmp.shape[-3],1/tmp.shape[-2]] if(self.nor)else [self.out_size[0]/tmp.shape[-3],self.out_size[1]/tmp.shape[-2]]
      if(self.out_size): tmp = tf.image.resize(tmp,self.out_size,'nearest')
      if(tmp.shape[-3]>self.max_size[0]):
        tmp = tf.image.resize(tmp,[self.max_size[0],tmp.shape[-2]],'nearest')
      if(tmp.shape[-2]>self.max_size[1]):
        tmp = tf.image.resize(tmp,[tmp.shape[-3],self.max_size[1]],'nearest')
      img_list.append(tf.reshape(tmp,[1]+tmp.shape))
      # read mask
      tmp = tf.image.decode_image(tf.io.read_file(os.path.join(self.pixeltraindir,mdir)))
      if(self.out_size): tmp = tf.image.resize(tmp,self.out_size,'nearest')
      if(tmp.shape[-3]>self.max_size[0]):
        tmp = tf.image.resize(tmp,[self.max_size[0],tmp.shape[-2]],'nearest')
      if(tmp.shape[-2]>self.max_size[1]):
        tmp = tf.image.resize(tmp,[tmp.shape[-3],self.max_size[1]],'nearest')
      y_list.append({
        'mask':tmp,
        'gt':txt_helper(os.path.join(self.txttraindir,'poly_gt_'+os.path.splitext(mdir)[0]+'.txt'),float,coe),
        })


    if(self.out_format=='tensor'):
      img_list = tf.convert_to_tensor(img_list)

    self.train_conter = cur_conter
    return img_list, y_list

  def read_test_batch(self, batch_size=10):
    img_names = self.test_img_names
    cur_conter, slice_a, slice_b = self.find_slice(self.test_conter,self.total_test,batch_size)
    img_list = []
    y_list = []
    boxs = []
    dirs = (img_names[slice_a] + img_names[slice_b]) if(slice_b)else img_names[slice_a]

    for mdir in dirs:
      tmp = tf.image.decode_image(tf.io.read_file(os.path.join(self.xtestdir,mdir)))
      coe = [1/tmp.shape[-3],1/tmp.shape[-2]] if(self.nor)else [self.out_size[0]/tmp.shape[-3],self.out_size[1]/tmp.shape[-2]]
      if(self.out_size): tmp = tf.image.resize(tmp,self.out_size,'nearest')
      if(tmp.shape[-3]>self.max_size[0]):
        tmp = tf.image.resize(tmp,[self.max_size[0],tmp.shape[-2]],'nearest')
      if(tmp.shape[-2]>self.max_size[1]):
        tmp = tf.image.resize(tmp,[tmp.shape[-3],self.max_size[1]],'nearest')
      img_list.append(tf.reshape(tmp,[1]+tmp.shape))
      
      tmp = tf.image.decode_image(tf.io.read_file(os.path.join(self.pixeltestdir,mdir)))
      if(self.out_size): tmp = tf.image.resize(tmp,self.out_size,'nearest')
      if(tmp.shape[-3]>self.max_size[0]):
        tmp = tf.image.resize(tmp,[self.max_size[0],tmp.shape[-2]],'nearest')
      if(tmp.shape[-2]>self.max_size[1]):
        tmp = tf.image.resize(tmp,[tmp.shape[-3],self.max_size[1]],'nearest')
      y_list.append({
        'mask':tmp,
        'gt':txt_helper(os.path.join(self.txttestdir,'poly_gt_'+os.path.splitext(mdir)[0]+'.txt'),float,coe),
        })

    if(self.out_format=='tensor'):
      img_list = tf.convert_to_tensor(img_list)
        
    self.test_conter = cur_conter
    return img_list, y_list

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



