import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
from lib.trainer import Trainer

if __name__ == "__main__":
  bx_choose = "top_k"
  model = Faster_RCNN(num_classes=2,bx_choose=bx_choose)
  loss = RCNNLoss(cfg,"TRAIN")
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  trainer = Trainer(isdebug=True,task_name=bx_choose)
  last_model = trainer.load(model)
  mydatalog = CTW(out_size=[512,512])

  if(last_model!=None):
    model = last_model
    mydatalog.setconter(trainer.data_count)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    )
  model(tf.zeros((1,512,512,3),dtype=tf.float32))
  
  for i in range(20):
    x_train, y_train = mydatalog.read_batch()
    trainer.fit(x_train,y_train,model,loss,optimizer)
    
    if(i%3==0):
      trainer.set_trainer(data_count=mydatalog._init_conter)
      trainer.save()

  print()

