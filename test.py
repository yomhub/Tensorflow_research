import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
from lib.model.config import cfg
from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss

if __name__ == "__main__":
  print(tf.version)
  model = Faster_RCNN(num_classes=2)
  loss = RCNNLoss(cfg,"TRAIN")
  mydatalog = CTW(out_size=[512,512])
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  # optimizer = tf.optimizers.SGD(learning_rate=0.001)
  model.compile(
    optimizer=optimizer,
    loss=loss,
    )
  y_pred = model(tf.zeros((1,512,512,3)))
  for x_train, y_train in mydatalog.pipline_entry():
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(model.trainable_variables)
      y_pred = model(x_train)
      loss_value = loss(y_train, y_pred)
    # grads = tape.gradient(loss_value, model.trainable_variables)
    g_rpn_cross_entropy = tape.gradient(loss.loss_detail["rpn_cross_entropy"], model.trainable_variables)
    g_rpn_loss_box = tape.gradient(loss.loss_detail["rpn_loss_box"], model.trainable_variables)
    g_cross_entropy = tape.gradient(loss.loss_detail["cross_entropy"], model.trainable_variables)
    g_loss_box = tape.gradient(loss.loss_detail["loss_box"], model.trainable_variables)
    # for itm in g_loss_box:
    #   nan_ind = tf.where(tf.logical_or(tf.math.is_nan(itm),tf.math.is_inf(itm)))
    #   if(nan_ind.shape[0]!=0):
    #     nan_obj = tf.gather(itm,nan_ind)
    optimizer.apply_gradients(zip(g_rpn_cross_entropy, model.trainable_variables))
    optimizer.apply_gradients(zip(g_rpn_loss_box, model.trainable_variables))
    print()

  # x_train, y_train = mydatalog.read_batch()
  # model.fit(
  #   x_train,  # input
  #   y_train,  # output
  #   batch_size=50,
  #   # verbose=0,  # Suppress chatty output; use Tensorboard instead
  #   epochs=1,
  #   # validation_data=(x_test, y_test),
  #   # callbacks=[
  #   #     tf.keras.callbacks.TensorBoard(run_dir),  # log metrics
  #   #     hp.KerasCallback(run_dir, hparams),  # log hparams
  #   # ],
  # )
  print("1")