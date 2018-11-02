

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.framework as framework
import module.config as config
from functools import partial

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate

def encoderNdecoder(
        images,
        out_channels=5,
        views=12,
        normalizer_fn=tf_layers.batch_norm,
        activation=tf.nn.leaky_relu):
    """
    images:n*h*x*c

    Returns:

    * results: N * 12 * 256 * 256 * 5
    """
    with tf.name_scope("model"):    
        images=tf.reshape(images,[-1,256,256,2])
        #images = tf.cast(images, tf.float32)
        with tf.variable_scope("encoder"):
            with framework.arg_scope([tf_layers.conv2d],
                                    kernel_size=4, stride=2, normalizer_fn=normalizer_fn,
                                    activation_fn=tf.nn.leaky_relu, padding="same"):
                e1 = tf_layers.conv2d(images, num_outputs=64)  # 256 x 256 x 64

                #e1 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                e2 = tf_layers.conv2d(e1, num_outputs=128)  # 64x64x128
                # e2 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                e3 = tf_layers.conv2d(e2, num_outputs=256)  # 32x32x256
                # e3 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                e4 = tf_layers.conv2d(e3, num_outputs=512)  # 16x16x512
                tf.add_to_collection('checkpoints',e4)
                # e4 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                e5 = tf_layers.conv2d(e4, num_outputs=512)  # 8x8x512
                # e5 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                e6 = tf_layers.conv2d(e5, num_outputs=512)  # 4X4X512
                # e6 = tf_layers.max_pool2d(net, [2, 2], scope='pool1')

                encoded = tf_layers.conv2d(e6, num_outputs=512)  # 2X2X512
                tf.add_to_collection('checkpoints',encoded)

        # vt=[None]*views #view
        va = []
        with tf.name_scope("decoders"):
            for count in range(views):
                with tf.variable_scope("decoder_{}".format(count)):
                    d6 = tf_layers.dropout(upsample(encoded, 512))  # 4X4x512
                    d5 = tf_layers.dropout(
                        upsample(tf.concat([d6, e6], 3), 512))  # 8X8X512
                    d4 = upsample(tf.concat([d5, e5], 3), 512)  # 16x16x512
                    tf.add_to_collection('checkpoints',d4)
                    d3 = upsample(tf.concat([d4, e4], 3), 256)  # 32x32x256
                    d2 = upsample(tf.concat([d3, e3], 3), 128)  # 64x64x128
                    tf.add_to_collection('checkpoints',d2)
                    d1 = upsample(tf.concat([d2, e2], 3), 64)  # 128x128x64
                    tf.add_to_collection('checkpoints',d1)
                    #tf.add_to_collection('checkpoints',d4)
                    decoded = upsample(
                        tf.concat(
                            [
                                d1,
                                e1],
                            3),
                        out_channels,
                        activation_fn=tf.nn.tanh,
                        normalizer_fn=tf_layers.batch_norm)  # 256x256x5

                    decoded = tf.nn.l2_normalize(
                        decoded,
                        axis=[1, 2, 3],
                        epsilon=1e-12,
                        name=None
                    )
                    va.append(decoded)

        # height = images.get_shape()[1].value
        # width = images.get_shape()[2].value
        # results = tf.reshape(tf.transpose(tf.stack(vt), [1,0,2,3,4]), [-1, height, width,out_channels])
        results = tf.stack(
            (va[0],va[1],va[2],va[3],va[4],va[5],
            va[6],va[7],va[8],va[9],va[10],va[11]),axis=-1)
        results = tf.transpose(results, [0, 4, 1, 2, 3])

        return results


def upsample(
        x,
        n_channels,
        kernel=4,
        stride=2,
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=tf_layers.batch_norm):
    """
    x is encoded
    """
    h_new = (x.get_shape()[1].value) * stride
    w_new = (x.get_shape()[2].value) * stride
    up = tf.image.resize_nearest_neighbor(x, [h_new, w_new])

    return tf_layers.conv2d(
        up,
        num_outputs=n_channels,
        kernel_size=kernel,
        stride=1,
        normalizer_fn=normalizer_fn,
        activation_fn=activation_fn)


        
    
    




########################################
########################################
########################################

def _pretty_print(var_names):
    '''
    Pretty name var names nicely.
    '''
    print("Encoder VARS \n\n")
    for var in var_names:
        if "encoder" in var:
            print(var)
        else:
            break

    print("Decoder VARS \n\n")
    for var in var_names:
        if "decoder" in var:
            print(var)

def keras_model(inputs=tf.keras.Input(shape=(None,256,256,2), dtype=tf.float32)):
    results = encoderNdecoder(inputs,out_channels=5,
        views=12)
    return tf.keras.Model(inputs=inputs, outputs=results)

def test():

    dummy_sketch = np.random.randn(1, 256, 256, 3) * 50 + 255
    dummy_sketch.astype(np.float32)
    results = encoderNdecoder(dummy_sketch)
    print(results.shape)
    _pretty_print([x.name for x in tf.global_variables()])

encoderNdecoder_lite = partial(encoderNdecoder, out_channels=5,
        views=12,
        normalizer_fn=tf_layers.batch_norm,
        activation=tf.nn.leaky_relu)
encoderNdecoder_lite= tf.contrib.layers.recompute_grad(encoderNdecoder_lite)

if __name__ == "__main__":
    test()
