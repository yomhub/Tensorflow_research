import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

__DEF_LINE_STY = [
    'solid',    # _____
    'dotted',   # .......
    'dashdot',  # __.__.__.
    'dashed',   # __ __ __
  ]
__DEF_COLORS = [
  'r','b','g','c','y','m']

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

def resize_visualize_helper(img,model,gtbox=None,mask=None):
  """
    Helper for feature part in unet in resize task.
    Args:
      img: input image
      gtbox: (N,4) with [y1,x1,y2,x2] in [0,1]
      mask: pixel mask
      module: model with output 
      {}
  """

  linewidth = 1.3
  plt.figure(figsize=(8,4))
  # fg = plt.figure(figsize=(8,4))
  plt.subplot(3,3,1,
  # figure=fg
  )
  divnum = 3*3-1
  base_scale = 32 # vgg net based scale
  if(type(img)!=list):img=[img]
  if(gtbox!=None and type(gtbox)!=list):gtbox=[gtbox]
  if(mask!=None and type(mask)!=list):mask=[mask]
  for j in range(divnum):
    plt.subplot(3,3,j+1,
      # figure=fg
    )
    plt.xlabel('layer',
      # figure=fg
    )
    plt.ylabel('energy',
    # figure=fg
    )
    plt.title('Scaler {}/{}'.format(j+1,divnum),
    # figure=fg
    )
  dx=np.arange(5)
  for i in range(len(img)):
    coe_x = int(max(int(img[i].shape[-2]/base_scale),divnum)/divnum)
    coe_y = int(max(int(img[i].shape[-3]/base_scale),divnum)/divnum)
    
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

      # if(dx==None):dx=np.arange(len(rt['ftlist']))
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

      plt.subplot(3,3,j+1,
        # figure=fg
      )
      plt.plot(dx,dmean,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[0],
        label='mean',
        # figure=fg
        )
      plt.plot(dx,dmin,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[1],
        label='min',
        # figure=fg
        )
      plt.plot(dx,dmax,
        color=__DEF_COLORS[i%len(__DEF_COLORS)],
        linewidth=linewidth,
        linestyle=__DEF_LINE_STY[2],
        label='max',
        # figure=fg
        )
  plt.show()
  plt.savefig('logfig.png')
  # fg.show()
  print("")

def sequence_visualize_helper(img,model,gtbox=None,mask=None):
  """
    Helper for feature part in unet in sequence task.
    Args:
      img: list of image sequence
      gtbox: (N,4) with [y1,x1,y2,x2] in [0,1]
      mask: pixel mask
      module: model with output 
      {}
  """
  def polygon_under_graph(xlist, ylist):
    '''
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    '''
    return [(xlist[0], 0.)] + list(zip(xlist, ylist)) + [(xlist[-1], 0.)]
  def cc(arg):
    '''
    Shorthand to convert 'named' colors to rgba format at 60% opacity.
    '''
    return mcolors.to_rgba(arg, alpha=0.6)
  fig_min = plt.figure(num='min')
  fig_max = plt.figure(num='max')
  fig_mean = plt.figure(num='mean')
  # mean, min, max
  ax_min = fig_min.gca(projection='3d')
  ax_max = fig_max.gca(projection='3d')
  ax_mean = fig_mean.gca(projection='3d')
  # ax1 = fig.add_subplot(1,3,1,projection='3d')
  # ax2 = fig.add_subplot(1,3,2,projection='3d')
  # ax3 = fig.add_subplot(1,3,3,projection='3d')
  linewidth = 1.3

  if(type(img)!=list):img=[img]
  if(gtbox!=None and type(gtbox)!=list):gtbox=[gtbox]
  if(mask!=None and type(mask)!=list):mask=[mask]

  dmin = []
  dmean = []
  dmax = []
  for i in range(len(img)):
    rt = model(img[i])
    mp = tf.cast(rt['scr'][:,:,:,1]>rt['scr'][:,:,:,0],tf.float32)
    mp = tf.reshape(mp,mp.shape+[1])
    mp = tf.broadcast_to(mp,mp.shape[:-1]+[3])
    mp = tf.concat([
      mp,
      # tf.broadcast_to(rt['mask'],rt['mask'].shape[:-1]+[3]),
      tf.image.resize(img[i],mp.shape[-3:-1])/255.0],
      axis=2)
    tf.summary.image(
      name = 'Score|Edg|Img image.',
      data = mp,step = i,
      max_outputs=20
      )
    tf.keras.preprocessing.image.save_img(
      path='fg{}.jpg'.format(i),
      x=tf.reshape(mp,mp.shape[1:]).numpy(),
      # scale=False,
    )
    tmp_dmin = []
    tmp_dmax = []
    tmp_dmean = []
    for j in range(len(rt['ftlist'])):  
      tmp_dmin += [tf.reduce_min(rt['ftlist'][j]).numpy()]
      tmp_dmax += [tf.reduce_max(rt['ftlist'][j]).numpy()]
      tmp_dmean += [tf.reduce_mean(rt['ftlist'][j]).numpy()]
    dmin += [tmp_dmin]
    dmax += [tmp_dmax]
    dmean += [tmp_dmean]

  # convert d[img][layer] to d[layer][img]
  dmin = np.asarray(dmin).transpose((1,0))
  dmax = np.asarray(dmax).transpose((1,0))
  dmean = np.asarray(dmean).transpose((1,0))

  zs = range(dmin.shape[0]) # layers num
  xs = np.arange(len(img))
  verts_min = []
  verts_max = []
  verts_mean = []
  cols = []
  for i in zs:
    verts_min.append(polygon_under_graph(xs,dmin[i]))
    verts_max.append(polygon_under_graph(xs,dmax[i]))
    verts_mean.append(polygon_under_graph(xs,dmean[i]))
    cols += [cc(__DEF_COLORS[i%len(__DEF_COLORS)])] 

  poly_min = PolyCollection(verts_min, facecolors=cols)
  poly_max = PolyCollection(verts_max, facecolors=cols)
  poly_mean = PolyCollection(verts_mean, facecolors=cols)
  
  ax_min.add_collection3d(poly_min, zs=zs, zdir='y')
  ax_min.set_xlabel('Images')
  ax_min.set_xlim(0, len(img))
  ax_min.set_ylabel('Layers')
  ax_min.set_ylim(0, dmin.shape[0])
  ax_min.set_zlabel('Mean')
  ax_min.set_zlim(dmin.min()-1.0, dmin.max()+1.0)
  ax_max.add_collection3d(poly_max, zs=zs, zdir='y')
  ax_max.set_xlabel('Images')
  ax_max.set_xlim(0, len(img))
  ax_max.set_ylabel('Layers')
  ax_max.set_ylim(0, dmin.shape[0])
  ax_max.set_zlabel('Max')
  ax_max.set_zlim(dmax.min()-1.0, dmax.max()+1.0)
  ax_mean.add_collection3d(poly_mean, zs=zs, zdir='y')
  ax_mean.set_xlabel('Images')
  ax_mean.set_xlim(0, len(img))
  ax_mean.set_ylabel('Layers')
  ax_mean.set_ylim(0, dmin.shape[0])
  ax_mean.set_zlabel('Mean')
  ax_mean.set_zlim(dmean.min()-1.0, dmean.max()+1.0)

  # plt.show()
  # plt.savefig('logfig.png')
  # fg.show()
  print("")