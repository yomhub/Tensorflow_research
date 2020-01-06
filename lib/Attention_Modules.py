import os, sys
import tensorflow as tf
import numpy as np

class L3Conv(tf.keras.Model):
    def __init__(self, chanel_num):
        self.chn = chanel_num
        super(L3Conv, self).__init__(name='L3Conv')

    def build(self, input_shape):

        self.conv_l1 = tf.keras.layers.Conv2D(
            self.chn, kernel_size=(3, 3), name="ConvL1_c"+str(self.chn))
        self.conv_l2 = tf.keras.layers.Conv2D(
            self.chn, kernel_size=(3, 3), name="ConvL2_c"+str(self.chn))
        self.conv_l3 = tf.keras.layers.Conv2D(
            self.chn, kernel_size=(3, 3), name="ConvL3_c"+str(self.chn))
        # Padding==Same:
        # H = H1 * stride
        # Padding==Valid:
        # H = (H1-1) * stride + HF]
        self.pad = tf.keras.layers.ZeroPadding2D(
            ((1,1),(1,1))
            )

    def call(self, inputs):
        cl1 = self.conv_l1(inputs)
        cl2 = self.conv_l2(cl1)
        cl3 = self.pad(self.conv_l2(cl2))

        return tf.keras.layers.Add(name="conv_out")([cl1, cl3])


class Vertical_Attention_Module(tf.keras.Model):
    def __init__(self):
        super(Vertical_Attention_Module, self).__init__(name='VAM')

    def build(self, input_shape):

        self.conv_l1 = tf.keras.layers.Conv2D(input_shape[-1],
                                              kernel_size=(1, 3),
                                              name=self.name + "conv_1_3"
                                              )
        self.conv_l2 = tf.keras.layers.Conv2D(input_shape[-1],
                                              kernel_size=(1, 1),
                                              name=self.name + "conv_1_1"
                                              )

    def call(self, inputs):

        self.inputs = inputs
        cv1 = self.conv_l1(inputs)
        cv2 = self.conv_l2(cv1)
        cvo = tf.keras.layers.Add(name="conv_out")([cv1, cv2])
        ex = tf.keras.backend.exp(cvo)
        b = tf.keras.backend.exp(
            cvo) / tf.keras.backend.cumsum(tf.keras.backend.exp(cvo))

        return tf.keras.layers.dot(inputs, b)

class Bidirectional_LSTM(tf.keras.Model):

    def __init__(self, units_num):
        self.units = units_num
        super(Bidirectional_LSTM, self).__init__(name='BLSTM')

    def build(self, input_shape):
        forward_layer   = tf.keras.layers.LSTM(self.units, return_sequences=True)
        backard_layer   = tf.keras.layers.LSTM(self.units, activation='relu', return_sequences=True,
                            go_backwards=True)
        self.glstm      = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backard_layer,
                                input_shape=(5, 10))

    def call(self, inputs):
        return self.glstm(inputs)


if __name__ == "__main__":
    testraay = np.array([[[[1, 1]]]])
    l3f = L3Conv(256)
    vam = Vertical_Attention_Module()
    _ = vam(tf.zeros([1, 254, 254, 256]))
    # _ = l3f(tf.zeros([1, 256, 256, 3]))
    # vam.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'])
    print(vam.summary())
