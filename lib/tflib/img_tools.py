import tensorflow as tf
import numpy as np

@tf.function
def overlap_tf(xbboxs,ybboxs):
  """
  Inputs:
    xbboxs: tensor wirh shape (M,4)
    ybboxs: tensor wirh shape (N,4)
      where 4 is (y1,x1,y2,x2)
  Assert:
    xbboxs.shape[-1]==ybboxs.shape[-1]==4
  Output:
    scores: shape (M,N)
  """
  assert(xbboxs.shape[-1]==4 and ybboxs.shape[-1]==4)
  xbboxs_t = tf.transpose(
    tf.reshape(xbboxs,[1,xbboxs.shape[0],xbboxs.shape[1]]),
    [1,0,2])
  ybboxs_t = tf.reshape(ybboxs,[1,ybboxs.shape[0],ybboxs.shape[1]])
  # shape (M,N)
  lap_y = tf.minimum(xbboxs_t[:,:,2],ybboxs_t[:,:,2])-tf.maximum(xbboxs_t[:,:,0],ybboxs_t[:,:,0])+1.0
  lap_x = tf.minimum(xbboxs_t[:,:,3],ybboxs_t[:,:,3])-tf.maximum(xbboxs_t[:,:,1],ybboxs_t[:,:,1])+1.0
  area=lap_x*lap_y
  unlap = tf.where(
    tf.logical_and(lap_x>0,lap_y>0),
    tf.add(
      # xbox area
      (xbboxs_t[:,:,2]-xbboxs_t[:,:,0]+1.0)*(xbboxs_t[:,:,3]-xbboxs_t[:,:,1]+1.0),
      # ybox area
      (ybboxs_t[:,:,2]-ybboxs_t[:,:,0]+1.0)*(ybboxs_t[:,:,3]-ybboxs_t[:,:,1]+1.0)
    )-area
    ,0
    )
  score = tf.where(unlap>0,area/unlap,0)
  return score

# @tf.function
def check_inside(boxes,img_size):
  """
    Args:
      boxes: (N,4) with [y1,x1,y2,x2]
      img_size: [height,width]
    Return: Bool mask wirh (N) shape
  """
  cond = tf.logical_and(
    tf.logical_and(boxes[:,0]>=0.0 , boxes[:,0]<=img_size[0]),
    tf.logical_and(boxes[:,1]>=0.0 , boxes[:,1]<=img_size[1])
  )
  cond2 = tf.logical_and(
    tf.logical_and(boxes[:,2]>=0.0 , boxes[:,0]<=img_size[0]),
    tf.logical_and(boxes[:,3]>=0.0 , boxes[:,1]<=img_size[1])
  )
  return tf.logical_and(cond,cond2)

def random_gt_generate(img, gtbox, increment=10, multiple=None, max_box_pre=None):
  """
    Args:
      img: tensor with (N,h,w,c)
      gtbox: 
        list (N>=1) of tensor or single (N=1) tensor
        tensor shape (gt_box_num,5) where 5 is
        [label,y1,x1,y2,x2]
      increment: int, increase number pre img
      multiple: if multiple is given, function will multiply
        gt_box_num as increment pre img
      max_box_pre: int, max gtbox num pre img.
    Return:
      img (N,h,w,c), list of tensor
  """
  if(type(gtbox)!=list):
    gtbox=[gtbox]
  gtimg = img.numpy()
  imgh = gtimg.shape[1]
  imgw = gtimg.shape[2]
  box_list = []
  for i in range(len(gtbox)):
    boxes = gtbox[i].numpy()
    gt_num = boxes.shape[0]

    add_num = increment if (multiple==None) else int(gt_num*multiple)
    if(max_box_pre!=None):
      add_num = add_num if (add_num+gt_num) < max_box_pre else max_box_pre-gt_num
    if(add_num<=0):
      continue
    
    # onlu use box small then half original image
    low_half = np.where(boxes[:,2]-boxes[:,0]<imgh & boxes[:,3]-boxes[:,1]<imgw).reshape([-1])
    if(low_half.shape[0]==0):
      continue
    low_half = np.take(boxes,low_half)
    txt_img = []
    for singletext in low_half:
      txt_img.append(gtimg[i,int(singletext[1]):int(singletext[3]),int(singletext[2]):int(singletext[4]),:])

    incs = np.random.randint(0,low_half.shape[0],(add_num))
    gen_box = np.take(low_half,incs)
    

    # [-0.5,0.5]
    dx = (np.random.random_sample((add_num))-0.5)*imgw
    dy = (np.random.random_sample((add_num))-0.5)*imgh

    gen_box[:,1:4:2] = np.where(gen_box[:,1]+dy<0 | gen_box[:,3]+dy>imgh,gen_box[:,1:4:2]-dy,gen_box[:,1:4:2]+dy)
    gen_box[:,2:5:2] = np.where(gen_box[:,2]+dy<0 | gen_box[:,4]+dy>imgh,gen_box[:,2:5:2]-dx,gen_box[:,2:5:2]+dx)
    for j in range(incs.shape[0]):
      gtimg[i,int(gen_box[j,1]):int(gen_box[j,3]),int(gen_box[j,2]):int(gen_box[j,4]),:] = txt_img[incs[j]]
    gen_box = np.concatenate((boxes,gen_box),axis=0)
    box_list.append(tf.convert_to_tensor(gen_box,dtype=gtbox[i].dtype))
  
  gtimg = tf.convert_to_tensor(gtimg,dtype=img.dtype)
  return gtimg, box_list