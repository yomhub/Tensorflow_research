import tensorflow as tf

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