import os, sys
import tensorflow as tf
import numpy as np
from dataset.ctw import CTW
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss

if __name__ == "__main__":
  model = Faster_RCNN(num_classes=2)
  myloss = RCNNLoss(cfg,"TRAIN")
  mydatalog = CTW()
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=myloss,
    )
  x_train, y_train = mydatalog.read_batch()
  model.fit(
    x_train,  # input
    y_train,  # output
    batch_size=10,
    # verbose=0,  # Suppress chatty output; use Tensorboard instead
    epochs=5,
    # validation_data=(x_test, y_test),
    # callbacks=[
    #     tf.keras.callbacks.TensorBoard(run_dir),  # log metrics
    #     hp.KerasCallback(run_dir, hparams),  # log hparams
    # ],
  )
