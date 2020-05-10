# Resnet50    
|Block1|Block2|Block3|Block4|Block5|Block6|
|:---:|:---:|:---:|:---:|:---:|:---:|
|input_1 (None, None, None, 3)|
|conv1_pad (None, None, None, 3)|
|conv1_conv (None, None, None, 64)|
|conv1_bn (None, None, None, 64)|
|conv1_relu (None, None, None, 64)|
|pool1_pad (None, None, None, 64)|
|pool1_pool (None, None, None, 64)|
|============|============|============|============|============|============|
|conv2_block1_1_conv|conv2_block2_1_conv|conv2_block3_1_conv|
|conv2_block1_1_bn|conv2_block2_1_bn|conv2_block3_1_bn|
|conv2_block1_1_relu|conv2_block2_1_relu|conv2_block3_1_relu|
|conv2_block1_2_conv|conv2_block2_2_conv|conv2_block3_2_conv|
|conv2_block1_2_bn|conv2_block2_2_bn|conv2_block3_2_bn|
|conv2_block1_2_relu|conv2_block2_2_relu|conv2_block3_2_relu|
|conv2_block1_0_conv|conv2_block2_3_conv|conv2_block3_3_conv|
|conv2_block1_3_conv|conv2_block2_3_bn|conv2_block3_3_bn|
|conv2_block1_0_bn|conv2_block2_add|conv2_block3_add|
|conv2_block1_3_bn|conv2_block2_out|conv2_block3_out (None, None, None, 256)|
|conv2_block1_out|
|conv2_block1_add|
|============|============|============|============|============|============|
|conv3_block1_1_conv|conv3_block2_1_conv|conv3_block3_1_conv|conv3_block4_1_conv|
|conv3_block1_1_bn|conv3_block2_1_bn|conv3_block3_1_bn|conv3_block4_1_bn|
|conv3_block1_1_relu|conv3_block2_1_relu|conv3_block3_1_relu|conv3_block4_1_relu|
|conv3_block1_2_conv|conv3_block2_2_conv|conv3_block3_2_conv|conv3_block4_2_conv|
|conv3_block1_2_bn|conv3_block2_2_bn|conv3_block3_2_bn|conv3_block4_2_bn|
|conv3_block1_2_relu|conv3_block2_2_relu|conv3_block3_2_relu|conv3_block4_2_relu|
|conv3_block1_0_conv|conv3_block2_3_conv|conv3_block3_3_conv|conv3_block4_3_conv|
|conv3_block1_3_conv|conv3_block2_3_bn|conv3_block3_3_bn|conv3_block4_3_bn|
|conv3_block1_0_bn|conv3_block2_add|conv3_block3_add|conv3_block4_add|
|conv3_block1_3_bn|conv3_block2_out|conv3_block3_out|conv3_block4_out (None, None, None, 512)|
|conv3_block1_add|
|conv3_block1_out|
|============|============|============|============|============|============|
|conv4_block1_1_conv|conv4_block2_1_conv|conv4_block3_1_conv|conv4_block4_1_conv|conv4_block5_1_conv|conv4_block6_1_conv|
|conv4_block1_1_bn|conv4_block2_1_bn|conv4_block3_1_bn|conv4_block4_1_bn|conv4_block5_1_bn|conv4_block6_1_bn|
|conv4_block1_1_relu|conv4_block2_1_relu|conv4_block3_1_relu|conv4_block4_1_relu|conv4_block5_1_relu|conv4_block6_1_relu|
|conv4_block1_2_conv|conv4_block2_2_conv|conv4_block3_2_conv|conv4_block4_2_conv|conv4_block5_2_conv|conv4_block6_2_conv|
|conv4_block1_2_bn|conv4_block2_2_bn|conv4_block3_2_bn|conv4_block4_2_bn|conv4_block5_2_bn|conv4_block6_2_bn|
|conv4_block1_2_relu|conv4_block2_2_relu|conv4_block3_2_relu|conv4_block4_2_relu|conv4_block5_2_relu|conv4_block6_2_relu|
|conv4_block1_0_conv|conv4_block2_3_conv|conv4_block3_3_conv|conv4_block4_3_conv|conv4_block5_3_conv|conv4_block6_3_conv|
|conv4_block1_3_conv|conv4_block2_3_bn|conv4_block3_3_bn|conv4_block4_3_bn|conv4_block5_3_bn|conv4_block6_3_bn|
|conv4_block1_0_bn|conv4_block2_add|conv4_block3_add|conv4_block4_add|conv4_block5_add|conv4_block6_add|
|conv4_block1_3_bn|conv4_block2_out|conv4_block3_out|conv4_block4_out|conv4_block5_out|conv4_block6_out(None, None, None, 1024)|
|conv4_block1_add|
|conv4_block1_out|
|============|============|============|============|============|============|
|conv5_block1_1_conv|conv5_block2_1_conv|conv5_block3_1_conv|
|conv5_block1_1_bn|conv5_block2_1_bn|conv5_block3_1_bn|
|conv5_block1_1_relu|conv5_block2_1_relu|conv5_block3_1_relu|
|conv5_block1_2_conv|conv5_block2_2_conv|conv5_block3_2_conv|
|conv5_block1_2_bn|conv5_block2_2_bn|conv5_block3_2_bn|
|conv5_block1_2_relu|conv5_block2_2_relu|conv5_block3_2_relu|
|conv5_block1_0_conv|conv5_block2_3_conv|conv5_block3_3_conv|
|conv5_block1_3_conv|conv5_block2_3_bn|conv5_block3_3_bn|
|conv5_block1_0_bn|conv5_block2_add|conv5_block3_add|
|conv5_block1_3_bn|conv5_block2_out|conv5_block3_out (None, None, None, 2048)|
|conv5_block1_add|
|conv5_block1_out|


# Res net Code
```
    res50 = tf.keras.applications.ResNet50(
      # input_tensor=tf.keras.Input(shape=input_shape[-3:]),
      weights='imagenet', 
      include_top=False)
    self.max_scale_factor = 16
    self.feature_model = tf.keras.Model(
      inputs=res50.inputs,
      outputs=[
        res50.get_layer('pool1_pool').output,
        # 1/4, ch=64
        res50.get_layer('conv2_block3_out').output,
        # 1/4, ch=256
        res50.get_layer('conv3_block4_out').output,
        # 1/8, ch=512
        res50.get_layer('conv4_block6_out').output,
        # 1/16, ch=1024
        # res50.get_layer('conv5_block3_out').output
        # 1/32, ch=2048
      ],
      name='res50'
    )

```

# UP conv coed

```
    self.g1upc = tf.keras.layers.Conv2DTranspose(
      filters = g1_g3_chs_list[0], 
      kernel_size = (3, 3),
      strides = (3, 3),
      use_bias = False,
      activation = 'relu',
      name = 'g1_upconv',
      )
```















