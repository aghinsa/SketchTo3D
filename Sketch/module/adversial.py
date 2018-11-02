
import tensorflow as tf
import module.config
import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework

def discriminate(images):
    """
    input:[n,h,w,5]
    returns probs n,1
    """
    images=tf.reshape(images,[-1,256,256,5])
    with tf.variable_scope("discriminator", reuse= tf.AUTO_REUSE):
        with framework.arg_scope([layers.conv2d],kernel_size=4,stride=2,activation_fn=tf.nn.leaky_relu,
                normalizer_fn=tf.contrib.layers.batch_norm,padding="same"):
            net=layers.conv2d(images,num_outputs=64)
            net=layers.conv2d(net,num_outputs=128)
            net=layers.conv2d(net,num_outputs=256)
            tf.add_to_collection('checkpoints',net)
            net=layers.conv2d(net,num_outputs=512)
            net=layers.conv2d(net,num_outputs=512)
            net=layers.conv2d(net,num_outputs=512)
            tf.add_to_collection('checkpoints',net)
            net=layers.conv2d(net,num_outputs=512)
        
        probs=tf.reshape(net,[-1,2048])
        probs=layers.fully_connected(probs,num_outputs=2,activation_fn=tf.nn.sigmoid)
        
    
    return probs

    ##########################################
                
def test():

    dummy_sketch = np.random.randn(4,256, 256, 5) * 50 + 255
    dummy_sketch=dummy_sketch.astype(np.float32)
    results = discriminate(dummy_sketch)


if __name__ == "__main__":
    test()
