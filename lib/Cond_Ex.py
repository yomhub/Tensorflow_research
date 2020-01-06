# utf-8
# this module for Conditional Spatial Expansion 
# based on paper 
# Towards Robust Curve Text Detection With Conditional Spatial Expansion

import os, sys
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Cond_Pred(layers.Layer):
  """
  X: input feature (chs,)
  Y: possibility of direction (5,)
  [button,right,left,top,stable]->BRLTS
  Ho: output hidden state TO BRLT, (4,1)
  Hi: input hidden state FROM CENTER and BRLT, (1+4,1)
  Direction: BRLT case = 4
  """
  def __init__(self, x_chs=4,direction=4):
    super(Cond_Pred, self).__init__()
    rd_init = tf.random_normal_initializer()
    z_init = tf.zeros_initializer()
    self.direction = direction
    self.wc = tf.Variable(initial_value=rd_init(
      shape=(direction, x_chs+5*direction+20),
      dtype='float32'),
      trainable=True)
    self.bc = tf.Variable(initial_value=z_init(
      shape=(direction, 1),
      dtype='float32'),
      trainable=True)

    # W and B for gate value in BRLT
    self.wgci = tf.Variable(initial_value=rd_init(
      shape=(direction, x_chs+5*direction+20),
      dtype='float32'),
      trainable=True)
    self.bgci = tf.Variable(initial_value=z_init(
      shape=(direction, 1),
      dtype='float32'),
      trainable=True)

    # W and B for current g in BRLT
    self.wgcur = tf.Variable(initial_value=rd_init(
      shape=(direction, x_chs+5*direction+20),
      dtype='float32'),
      trainable=True)
    self.bgcur = tf.Variable(initial_value=z_init(
      shape=(direction, 1),
      dtype='float32'),
      trainable=True)

    # W and B for output gate in BRLT
    self.wgout = tf.Variable(initial_value=rd_init(
      shape=(direction, x_chs+5*direction+20),
      dtype='float32'),
      trainable=True)
    self.bgout = tf.Variable(initial_value=z_init(
      shape=(direction, 1),
      dtype='float32'),
      trainable=True)

    # B for output H 
    self.bhout = tf.Variable(initial_value=z_init(
      shape=(direction, 1),
      dtype='float32'),
      trainable=True)
      
    # W and B for Y in CENTER and BRLT
    # convert c (self.direction,1) to (5,1)
    self.wyout = tf.Variable(initial_value=rd_init(
      shape=(5, direction),
      dtype='float32'),
      trainable=True)
    self.byout = tf.Variable(initial_value=z_init(
      shape=(5, 1),
      dtype='float32'),
      trainable=True)

    # self.cin = tf.Variable(initial_value=z_init(
    #   shape=(4, direction),
    #   dtype='float32'),
    #   trainable=True)

    
  def call(self, inputs):
    """
    Inputs
    Xin: input feature (chs,1)
    Hin = (hcenter,hb,hr,hl,ht), shape (5,self.direction)
    Cin = (cbin,crin,clin,ctin), shape (4,self.direction)
    Yin = (yb,yr,yl,yt) shape (4,5)

    Hout shape: (self.direction)
    Y possibility in direction (center,button,right,left,top), shape: (5)

    Outputs
    Yout = (5,1)
    Hout = (self.direction,1)
    Cout = (self.direction,self.direction)
    """

    xin, yin, hin, cin = inputs
    assert(hin.shape==(5,self.direction))
    assert(cin.shape==(4,self.direction))
    # s shape (chx+5*self.direction+4*5, 1)
    s = tf.concat([
      tf.reshape(xin,[-1]),
      tf.reshape(hin,[-1]),
      tf.reshape(yin,[-1]),
      ], 
      0)
    s = tf.reshape(s,[-1,1])
    # current candidate state, shape (self.direction,1)
    cur_c = tf.tanh(tf.matmul(self.wc,s)+self.bc)
    # gcin, shape (self.direction,1)
    gcin = tf.sigmoid(tf.matmul(self.wgci,s)+self.bgci)
    tmp = tf.zeros(cur_c.shape)
    for i in range(gcin.shape[0]):
      tmp += gcin[i]*tf.reduce_sum(cin[i])
    # gcur shape (self.direction,1)
    gcur = tf.sigmoid(tf.matmul(self.wgcur,s)+self.bgcur)
    # c shape same as gcur (self.direction,self.direction)
    c = tf.keras.utils.normalize(tmp+(gcur*cur_c))

    # gout shape (self.direction,1)
    gout = tf.sigmoid(tf.matmul(self.wgout,s)+self.bgout)
    # hout shape (self.direction,1)
    hout = tf.tanh(c)*gout+self.bhout

    y = tf.nn.softmax(tf.matmul(self.wyout,c)+self.byout)
    return y,hout,c

@tf.function
def _roi_loss():
  
  pass

class CSE(tf.keras.Model):
  def __init__(self,
  feature_layer_name='vgg16',
  proposal_window_size=[3,3],
  max_feature_size=[30,30]
  ):
    super(CSE, self).__init__()
    # self.name='Faster_RCNN'
    self.pw_size=proposal_window_size
    self._predictions={}
    self._loss_function=_roi_loss()
    if(feature_layer_name=='vgg16'):
      self.feature_layer_name=feature_layer_name
      self.cond_pred_layer=Cond_Pred(x_chs=512)
    elif(feature_layer_name.lower()=='resnet'):
      self.feature_layer_name='resnet'
    else:
      self.feature_layer_name='vgg16'
    if(type(max_feature_size)==list):
      self.max_feature_size=max_feature_size
    else:
      self.max_feature_size=[int(max_feature_size),int(max_feature_size)]
  
  def build(self, 
  input_shape,
  ):
    if(self.feature_layer_name=='resnet'):
      rn=tf.keras.applications.ResNet101V2()
      self.feature_model = tf.keras.models.Sequential([
        # vgg16.get_layer("input_1"),
        rn.get_layer("conv1_pad"), rn.get_layer("conv1_conv"), rn.get_layer("pool1_pad"), rn.get_layer("pool1_pool"),
        rn.get_layer("block2_conv1"), rn.get_layer("block2_conv2"), rn.get_layer("block2_pool"),
        rn.get_layer("block3_conv1"), rn.get_layer("block3_conv2"), rn.get_layer("block3_conv3"), 
        rn.get_layer("block3_pool"),
        rn.get_layer("block4_conv1"), rn.get_layer("block4_conv2"), rn.get_layer("block4_conv3"),
        rn.get_layer("block4_pool"),
        rn.get_layer("block5_conv1"), rn.get_layer("block5_conv2"), rn.get_layer("block5_conv3"),
        rn.get_layer("block5_pool"),
      ],
      name=self.feature_layer_name
      )
    else:
      # default VGG16
      vgg16=tf.keras.applications.VGG16(weights='imagenet', include_top=False)
      self.feature_model = tf.keras.models.Sequential([
        # tf.keras.Input((1024,1024,3)),
        # vgg16.get_layer("input_1"),
        # Original size
        vgg16.get_layer("block1_conv1"), vgg16.get_layer("block1_conv2"), vgg16.get_layer("block1_pool"),
        # Original size / 2
        vgg16.get_layer("block2_conv1"), vgg16.get_layer("block2_conv2"), vgg16.get_layer("block2_pool"),
        # Original size / 4
        vgg16.get_layer("block3_conv1"), vgg16.get_layer("block3_conv2"), vgg16.get_layer("block3_conv3"), 
        # Original size / 4
        vgg16.get_layer("block3_pool"),
        # Original size / 8
        vgg16.get_layer("block4_conv1"), vgg16.get_layer("block4_conv2"), vgg16.get_layer("block4_conv3"),
        # Original size / 8
        vgg16.get_layer("block4_pool"),
        # Original size / 16
        vgg16.get_layer("block5_conv1"), vgg16.get_layer("block5_conv2"), vgg16.get_layer("block5_conv3"),
        # Original size / 16
        # vgg16.get_layer("block5_pool"),
        # Original size / 32
      ],
      name=self.feature_layer_name
      )
  def call(self,inputs):
    """
    Features generator->Conditional Spatial Expansion
    input: image
    """
    feature = self.feature_model(inputs)
    # for bach in 
    return feature



if __name__ == "__main__":
  # test_Cond_Pred = Cond_Pred()

  # inp=[
  #   tf.zeros((4,1)),
  #   tf.zeros((4,5)),
  #   tf.zeros((5,4)),
  #   tf.zeros((4,4)),
  # ]
  # y,hout=test_Cond_Pred(inp)
  test_model = CSE()
  # RGB wth 256*256
  inp=tf.zeros((1,256,256,3))
  y = test_model(inp)
  print(y.shape)
  pass