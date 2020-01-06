import tensorflow as tf

@tf.function
def overlap_tf(xbboxs,ybboxs):
  """
  Inputs:
    xbboxs: tensor wirh shape (M,4)
    ybboxs: tensor wirh shape (N,4)
      where 4 is (y1,x1,y2,x2)
  Output:
    scores: shape (M,N)
  """
  xbboxs_t=tf.transpose(
    tf.reshape(xbboxs,[1,xbboxs.shape[0],xbboxs.shape[1]]),
    [1,0,2])
  ybboxs_t=tf.reshape(ybboxs,[1,ybboxs.shape[0],ybboxs.shape[1]])
  # shape (M,N)
  lap_y = tf.minimum(xbboxs_t[:,:,2],ybboxs_t[:,:,2])-tf.maximum(xbboxs_t[:,:,0],ybboxs_t[:,:,0])+1.0
  lap_x = tf.minimum(xbboxs_t[:,:,3],ybboxs_t[:,:,3])-tf.maximum(xbboxs_t[:,:,1],ybboxs_t[:,:,1])+1.0
  aria=lap_x*lap_y
  unlap = tf.where(
    tf.logical_and(lap_x>0,lap_y>0),
    tf.add(
      (xbboxs_t[:,:,2]-xbboxs_t[:,:,0]+1.0)*(xbboxs_t[:,:,3]-xbboxs_t[:,:,1]+1.0),
      (ybboxs_t[:,:,2]-ybboxs_t[:,:,0]+1.0)*(ybboxs_t[:,:,3]-ybboxs_t[:,:,1]+1.0)
    )-aria
    ,0
    )
  score = tf.where(unlap>0,aria/unlap,0)
  return score