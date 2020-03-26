import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def str2time(instr):
  ymd,hms=instr.split('-')
  return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]), int(hms[:2]), int(hms[2:4]), int(hms[4:6]))

def str2num(instr):
  return [int(s) for s in instr.split() if s.isdigit()]

def auto_scalar(dic_data, step=0, logname=None):
  if(type(dic_data)==list):
    if(logname==None):
      logname="auto_scalar"
    cont = 0
    for itm in dic_data:
      tf.summary.scalar(logname+"_list_{}".format(cont),itm,step=step)
      cont += 1
  elif(type(dic_data)==dict):
    for itname in dic_data:
      tf.summary.scalar(itname,dic_data[itname],step=step)
  else:
    if(logname==None):
      logname="auto_scalar"
    tf.summary.scalar(logname,dic_data,step=step)

def auto_image(img_data, name=None, step=0, max_outputs=None, description=None):
  """
    Args:  
      img_data: tensor with shape (h,w,3 or 1) or (N,h,w,3 or 1)
      name: log name
      step: int of step
      max_outputs: int of max_outputs
  """
  if(len(img_data)==3):
    img_data = tf.reshape(img_data,[1,]+img_data.shape)
  if(tf.reduce_max(img_data)>1.0):
    img_data = img_data / 256.0
  max_outputs = img_data.shape[0] if max_outputs==None else max_outputs
  name = "auto_image" if name==None else name
  tf.summary.image(name,img_data,step,max_outputs,description)

def auto_histogram(dic_data, step=0, logname=None):
  if(type(dic_data)==list):
    if(logname==None):
      logname="auto_scalar"
    cont = 0
    for itm in dic_data:
      tf.summary.histogram(logname+"_list_{}".format(cont),itm,step=step)
      cont += 1
  elif(type(dic_data)==dict):
    for itname in dic_data:
      tf.summary.histogram(itname,dic_data[itname],step=step)
  else:
    if(logname==None):
      logname="auto_scalar"
    tf.summary.histogram(logname,dic_data,step=step)

def save_image(img, savedir):
  if(len(img.shape)==4):
    img = tf.reshape(img,img.shape[1:])
  tf.io.write_file(savedir,tf.io.encode_jpeg(tf.cast(img,tf.uint8)))

def plt_draw_lines(funcs,xarry=None,cols=None,fig_size=None,fig_num=None,save_name=None):
  """

  """
  plt.figure(num=fig_num,figsize=fig_size)
  if(xarry==None):
    xarry=np.range(-5.0,5.0,100)
  if(type(funcs)!=list):
    funcs = [funcs]
  tcols = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
  line_sty = [
    '','--','-.','--.',':'
  ]
  for i in range(len(funcs)):
    plt.plot(xarry,funcs[i],tcols[i],linewidth=1.0,linestyle=line_sty[i])
  plt.show()

def rf_helper(net_list,ord_len,panding=True):
  """
    Args:
      net_list: list pramaters of network
        [kernel size, stride size]
      ord_num: int, coordinate range
      panding: True or False
    Print coordinate in each layer
  """
  if(type(net_list)!=list or len(net_list)<2):return
  panding = bool(panding)
  ord_len = int(ord_len)
  rf_st = [[1,1]]
  cod_table = np.arange(ord_len,dtype=np.int)
  cod_table = np.stack((cod_table,cod_table),axis=-1)
  cod_table = [cod_table.tolist()]

  for i in range(len(net_list)):
    rf,st = rf_st[i]
    ksize,strsize = net_list[i]
    crf = rf + (ksize-1)*st
    cst = st*strsize
    rf_st.append([crf,cst])

    p_harf_k = int(ksize/2) if((ksize-int(ksize/2)*2)!=0)else int(ksize/2)-1
    harf_k = ksize - 1 - p_harf_k
    max_cod = len(cod_table[i])-1
    stp = 0 if panding else p_harf_k
    edp = max_cod if panding else max_cod - harf_k
    tmp = []
    while(stp<edp):
      c_ctp = max(0,stp-p_harf_k)
      c_edp = min(max_cod,stp + harf_k)
      tmp.append([cod_table[i][c_ctp][0],cod_table[i][c_edp][1]])
      stp+=strsize
    cod_table.append(tmp)
  return rf_st,cod_table