import os, sys
import tensorflow as tf
import numpy as np
from tensorboard.plugins.hparams import api as hp
import Dataset.svt as svt

logs_path = "\\tmp\\logs_file"
task_name = "\\AttentionFilter_"

HP_NUM_CHS = hp.HParam('num_chs', hp.Discrete([128, 256, 512]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1000, 5000]))
METRIC_ACCURACY = 'accuracy'
tf.keras.callbacks.TensorBoard()
logT = open(logs_path+"log.txt", 'w', encoding='utf8')
logT.write(task_name+"\n")
_xtrain,_ytrain,_xtest,_ytest = svt._read_swt(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../Dataset/svt"))


def train_test_model(run_dir,hparams, x_train, y_train, x_test, y_test):

    # CNN = cnnl1 + cnnl3(cnnl2(cnnl1))
    conv_l1  =   tf.keras.layers.Conv2D(hparams[HP_NUM_CHS],kernel_size=(3,3),name="ConvL1_c"+str(hparams[HP_NUM_CHS]))
    conv_l2  =   tf.keras.layers.Conv2D(hparams[HP_NUM_CHS],kernel_size=(3,3),name="ConvL2_c"+str(hparams[HP_NUM_CHS]))(conv_l1)
    conv_l3  =   tf.keras.layers.Conv2D(hparams[HP_NUM_CHS],kernel_size=(3,3),name="ConvL3_c"+str(hparams[HP_NUM_CHS]))(conv_l2)
    conv_add =   tf.keras.layers.Add(name="conv_out")([conv_l1, conv_l3])


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hparams[HP_NUM_UNITS], activation=tf.nn.relu, name="Dense_0_"+str(hparams[HP_NUM_UNITS])),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(
            hparams[HP_NUM_UNITS], activation=tf.nn.relu, name="Dense_1_"+str(hparams[HP_NUM_UNITS])),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ],
        name="model_AtenFil_"+str(hparams[HP_NUM_CHS])
    )
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Run with 1 epoch to speed things up for demo purposes
    model.fit(
        x_train,  # input
        y_train,  # output
        batch_size=hparams[HP_BATCH_SIZE],
        epochs=5,
        callbacks=[
          tf.keras.callbacks.TensorBoard(run_dir),  # log metrics
          hp.KerasCallback(run_dir, hparams),  # log hparams
        ],
    )
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy