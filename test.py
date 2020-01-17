import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.trainer import Trainer

if __name__ == "__main__":
  print(tf.version)
  model = Faster_RCNN(num_classes=2,bx_choose="nms")
  loss = RCNNLoss(cfg,"TRAIN")
  mydatalog = CTW(out_size=[512,512])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  trainer = Trainer()
  # optimizer = tf.optimizers.SGD(learning_rate=0.001)
  model.compile(
    optimizer=optimizer,
    loss=loss,
    )
  y_pred = model(tf.zeros((1,512,512,3)))

  x_train, y_train = mydatalog.read_batch()
  trainer.fit(x_train,y_train,model,loss,optimizer)
