import tensorflow as tf
from datetime import datetime

def _str2time(instr):
  ymd,hms=instr.split('-')
  return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]), int(hms[:2]), int(hms[2:4]), int(hms[4:6]))

def _str2num(instr):
  return [int(s) for s in instr.split() if s.isdigit()]

def _auto_scalar(dic_data, step=0, logname=None):
  if(type(dic_data)==list):
    if(logname==None):
      logname="_auto_scalar"
    cont = 0
    for itm in dic_data:
      tf.summary.scalar(logname+"_list_{}".format(cont),itm,step=step)
      cont += 1
  elif(type(dic_data)==dict):
    for itname in dic_data:
      tf.summary.scalar(itname,dic_data[itname],step=step)
  else:
    if(logname==None):
      logname="_auto_scalar"
    tf.summary.scalar(logname,dic_data,step=step)

def _auto_image(img_data, name=None, step=0, max_outputs=None, description=None):
  if(len(img_data)==3):
    img_data = tf.reshape(img_data,[1,]+img_data.shape)
  if(tf.reduce_max(img_data)>1.0):
    img_data = img_data / 256.0
  max_outputs = img_data.shape[0] if max_outputs==None else max_outputs
  name = "_auto_image" if name==None else name
  tf.summary.image(name,img_data,step,max_outputs,description)

def _auto_histogram(dic_data, step=0, logname=None):
  if(type(dic_data)==list):
    if(logname==None):
      logname="_auto_scalar"
    cont = 0
    for itm in dic_data:
      tf.summary.histogram(logname+"_list_{}".format(cont),itm,step=step)
      cont += 1
  elif(type(dic_data)==dict):
    for itname in dic_data:
      tf.summary.histogram(itname,dic_data[itname],step=step)
  else:
    if(logname==None):
      logname="_auto_scalar"
    tf.summary.histogram(logname,dic_data,step=step)

def save_image(img, savedir):
  if(len(img.shape)==4):
    img = tf.reshape(img,img.shape[1:])
  tf.io.write_file(savedir,tf.io.encode_jpeg(tf.cast(img,tf.uint8)))