import tensorflow as tf
from trainer import Trainer

class FRCNNTrainer(Trainer):
  def __init__(self):
    Trainer.__init__(self)
  
  def train_action(self,x_single,y_single):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.model.trainable_variables)
      y_pred = self.model(x_single)
      loss_value = self.loss(y_single, y_pred)
    gt_prob = y_pred["rpn_cls_score1"]
    gt_prob = tf.reshape(gt_prob,[-1,gt_prob.shape[-1]])
    gt_prob = gt_prob[:,int(gt_prob.shape[-1]/2):]
    gt_prob = tf.reshape(gt_prob,[-1])
    gt_boxes = y_pred["rpn_bbox_pred1"]
    gt_boxes = tf.reshape(gt_boxes,[-1,4])
    gt_in = tf.reshape(tf.where(gt_prob>0.8),[-1])
    bbx = tf.gather(gt_boxes,gt_in)
    gtbox_num = int(gt_in.shape[0])