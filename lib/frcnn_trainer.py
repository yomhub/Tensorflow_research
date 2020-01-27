import tensorflow as tf
from trainer import Trainer

class FRCNNTrainer(Trainer):
  def __init__(self):
    Trainer.__init__(self)
  
  def train_action(self,x_single,y_single):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(model.trainable_variables)
      y_pred = model(x_train[step])
      loss_value = loss(y_train[step], y_pred)