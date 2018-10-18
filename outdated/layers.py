# import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.layers as tf_layers
#
# def encoder(images,normalizer_fn=tf_layers.batch_norm,activation=tf.nn.leaky_relu):
#     """
#     images:n*h*x*c
#     """
#
#     e1=tf_layers.conv2d(images,num_outputs=64,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     e2=tf_layers.conv2d(images,num_outputs=128,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     e3=tf_layers.conv2d(images,num_outputs=256,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     e4=tf_layers.conv2d(images,num_outputs=512,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     e5=tf_layers.conv2d(images,num_outputs=512,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     e6=tf_layers.conv2d(images,num_outputs=512,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#     encoded=tf_layers.conv2d(images,num_outputs=512,kernel_size=4,stride=2,normalizer_fn=normalizer_fn,activation_fn=tf.nn.leaky_relu)
#
#     num_images=images.get_shape()[0].value
#     #features=tf.reshape(encoded,[num_images,-1])
#
#     return encoded
#
# def upsample(x,n_channels,kernel=4,stride=2,activation_fn=tf.nn.leaky_relu,normalizer_fn=tf_layers.batch_norm):
#     """
#     x is encoded
#     """
#     h_new=(x.get_shape()[1].value)*stride
#     w_new=(x.get_shape()[2].value)*stride
#     up=tf.image.resize_nearest_neighbor(x,[h_new,w_new])
#
#     return tf_layers.conv2d(up,num_outputs=n_channels,kernel_size=kernel,stride=1,normalizer_fn=normalizer_fn,activation_fn=activation_fn)
#
# def decoder(encoded,out_channels):
#     d6=tf_layers.droput(upsample(encoded,512))
#     d5=tf_layers.droput(upsample(tf.concat([d6,e6],3),512))
#     d4=upsample(tf.concat([d5,e5],3),512)
#     d3=upsample(tf.concat([d4,e4],3),256)
#     d2=upsample(tf.concat([d3,e3],3),128)
#     d1=upsample(tf.concat([d2,e2],3),64)
#     decoded=upsample(tf.concat([d1,e1],3),out_channels)
#
#     return decoded
