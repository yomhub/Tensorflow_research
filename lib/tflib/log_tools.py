import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

__DEF_LINE_STY = [
    'solid',    # _____
    'dotted',   # .......
    'dashdot',  # __.__.__.
    'dashed',   # __ __ __
  ]
__DEF_COLORS = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

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

def plt_func_lines(funcs,xarry=None,cols=None,figure=None,fig_size=None,save_name=None):
  """
  Draw plt by function
  Args:
    fig_size: width, height
  """
  fg = plt.figure(figsize=fig_size) if(figure==None)else figure
  if(xarry==None):
    xarry=np.range(-5.0,5.0,100)
  if(type(funcs)!=list):
    funcs = [funcs]

  for i in range(len(funcs)):
    fg.plot(
      xarry,funcs[i](xarry),
      color=__DEF_COLORS[i%len(__DEF_COLORS)],
      linewidth=1.0,
      linestyle=__DEF_LINE_STY[i%len(__DEF_LINE_STY)])
  if(save_name!=None):fg.savefig(save_name)
  return fg

def plt_points_lines(points,xarry=None,xcfg=None,figure=None,fig_size=None,save_name=None):
  """
  Draw plt by function
  Args:
    points: 
      1D array (Ny,) with [y0,y1...]: 
        if xarry is [x0,x1...], use xarry.
        Or calculate xarry by xcfg
      2D array (2,Nyx) with [[y0,y1...],[x0,x1...]]
      List of 1D/2D arrays: 
        draw multi lines in figure
    xarry:
      1D array (Nx,) with [x0,x1...]
    xcfg:
      if points is 1D and xarry==None, (start,end) will be use
    
  """
  fg = plt.figure(figsize=fig_size) if(figure==None)else figure
  if(type(points)!=list):
    points = [points]

  for i in range(len(points)):
    if(len(points[i].shape)==1):
      if(xarry!=None and xarry.shape[0]==points[i].shape[0]): xs = xarry
      elif(xcfg!=None): xs = np.linspace(xcfg[0],xcfg[1],points[i].shape[0])
      ys = points[i]
    else:
      xs = points[i][1]
      ys = points[i][0]
    fg.plot(xs,ys,
      color=__DEF_COLORS[i%len(__DEF_COLORS)],
      linewidth=1.0,
      linestyle=__DEF_LINE_STY[i%len(__DEF_LINE_STY)])
  if(save_name!=None):fg.savefig(save_name)
  return fg

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

def visualize_helper(img,gtbox,mask,model):
  """
    Helper for feature part in unet.
    Args:
      img: input image
      gtbox: (N,4) with [y1,x1,y2,x2] in [0,1]
      mask: pixel mask
      module: model with output 
      {}
  """

  linewidth = 1.3
  fg = plt.figure(figsize=(8,4))
  fg.subplot(3,3,3)
  divnum = 3*3-1
  base_scale = 32 # vgg net based scale
  if(type(img)!=list):img=[img]
  if(type(gtbox)!=list):gtbox=[gtbox]
  if(type(mask)!=list):mask=[mask]
  for j in range(divnum):
    fg.subplot(3,3,j+1)
    fg.xlabel('layer')
    fg.ylabel('energy')
    fg.title('Scaler {}/{}'.format(j+1,divnum))
  dx = None
  for i in range(len(img)):
    coe_x = int(max(int(img.shape[-2]/base_scale),divnum)/divnum)
    coe_y = int(max(int(img.shape[-3]/base_scale),divnum)/divnum)
    
    for j in range(divnum):
      img_size = [coe_y*base_scale,coe_x*base_scale]
      tmp = tf.image.resize(img[i],img_size)
      rt = model(tmp)
      mp = tf.cast(rt['scr'][:,:,:,1]>rt['scr'][:,:,:,0],tf.float32)
      mp = tf.reshape(mp,mp.shape+[1])
      mp = tf.broadcast_to(mp,mp.shape[:-1]+[3])
      mp = tf.concat([mp,tf.image.resize(tmp,mp.shape[-3:-1])/255.0],axis=2)
      tf.summary.image(
        name = 'Score|Img image size {}.'.format(img_size),
        data = mp,step=0)

      if(dx==None):dx=np.arange(len(rt['ftlist']))
      dmin = []
      dmean = []
      dmax = []
      for o in range(len(rt['ftlist'])):
        dmin += [tf.reduce_min(rt['ftlist'][o]).numpy()]
        dmax += [tf.reduce_max(rt['ftlist'][o]).numpy()]
        dmean += [tf.reduce_mean(rt['ftlist'][o]).numpy()]

      dmin = np.asarray(dmin)
      dmean = np.asarray(dmean)
      dmax = np.asarray(dmax)

      fg.subplot(3,3,j+1)
      fg.plot(dx,dmean,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[0],
        label='mean')
      fg.plot(dx,dmin,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[1],
        label='min')
      fg.plot(dx,dmax,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[2],
        label='max')
fg.savefig('logfig.jpg')
