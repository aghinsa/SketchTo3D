
import module.config
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def network(inputs=tf.keras.Input(shape=(256,256,2))):
    views=12
    # image=tf.keras.layers.Input(shape=(256,256,2))
    with tf.name_scope("encoder"):
        
        net=layers.Conv2D(filters=64,kernel_size=4,
                strides=2,padding='same')(inputs) #256,256,64
        net=layers.LeakyRelu()(net)
        e1=layers.BatchNormalization(name='e1')(net)
        
        net=layers.Conv2D(filters=128,kernel_size=4,
                strides=2,padding='same')(e1) #256,256,128
        net=layers.LeakyRelu()(net)
        e2=layers.BatchNormalization(name='e2')(net)
        
        net=layers.Conv2D(filters=256,kernel_size=4,
                strides=2,padding='same')(e2) #256,256,256
        net=layers.LeakyRelu()(net)
        e3=layers.BatchNormalization(name='e3')(net)
        
        net=layers.Conv2D(filters=512,kernel_size=4,
                strides=2,padding='same')(e3) #256,256,512
        net=layers.LeakyRelu()(net)
        e4=layers.BatchNormalization(name='e4')(net)
        
        net=layers.Conv2D(filters=512,kernel_size=4,
                strides=2,padding='same')(e4) #256,256,512
        net=layers.LeakyRelu()(net)
        e5=layers.BatchNormalization(name='e5')(net)
        
        net=layers.Conv2D(filters=512,kernel_size=4,
                strides=2,padding='same')(e5) #256,256,512
        net=layers.LeakyRelu()(net)
        e6=layers.BatchNormalization(name='e6')(net)
        
        net=layers.Conv2D(filters=512,kernel_size=4,
                strides=2,padding='same')(e6) #256,256,512
        net=layers.LeakyRelu()(net)
        encoder_out=layers.BatchNormalization(name='encoded')(net)
        
    
    va=[] #view_array
    
    
    
    for view in range(views):
        with tf.name_scope("decoder_{}".format(view+1)):
            
            d6=layers.Conv2DTranspose(filters=512,kernel_size=4,strides=1)(net)
            d6=layers.LeakyRelu()(d6)
            d6=layers.BatchNormalization()(d6)
            d6=layers.Dropout(rate=0.5,)(d6)
        
            d5=layers.concatenate(inputs=[d6,e6],axis=-1)
            d5=layers.Conv2DTranspose(filters=512,kernel_size=4,strides=1)(d5)
            d5=layers.LeakyRelu()(d5)
            d5=layers.BatchNormalization()(d5)
            d5=layers.Dropout(rate=0.5)(d5)
            
            d4=layers.concatenate(inputs=[d5,e5],axis=-1)
            d4=layers.Conv2DTranspose(filters=512,kernel_size=4,strides=1)(d4)
            d4=layers.LeakyRelu()(d4)
            d4=layers.BatchNormalization()(d4)
            
            d3=layers.concatenate(inputs=[d4,e4],axis=-1)
            d3=layers.Conv2DTranspose(filters=256,kernel_size=4,strides=1)(d3)
            d3=layers.LeakyRelu()(d3)
            d3=layers.BatchNormalization()(d3)
            
            d2=layers.concatenate(inputs=[d3,e3],axis=-1)
            d2=layers.Conv2DTranspose(filters=128,kernel_size=4,strides=1)(d2)
            d2=layers.LeakyRelu()(d2)
            d2=layers.BatchNormalization()(d2)
            
            d1=layers.concatenate(inputs=[d2,e2],axis=-1)
            d1=layers.Conv2DTranspose(filters=64,kernel_size=4,strides=1)(d4)
            d1=layers.LeakyRelu()(d4)
            d1=layers.BatchNormalization()(d4)
            
            decoded=layers.concatenate(inputs=[d1,e1],axis=-1)
            decoded=layers.Conv2DTranspose(filters=5,kernel_size=4,strides=1,
                    activation=tf.keras.activations.tanh)(decoded)
            decoded=tf.keras.backend.l2_normalize (decoded,axis=[1,2,3])

            va.append(decoded)
            
        results = tf.stack(
            (va[0],
             va[1],
                va[2],
                va[3],
                va[4],
                va[5],
                va[6],
                va[7],
                va[8],
                va[9],
                va[10],
                va[11]),
            axis=-1)
        results = tf.transpose(results, [0, 4, 1, 2, 3])
            
    
    return results
                                    
    
    
###########################################################3
import numpy as np

images=np.random.rand((2,256,256,2))
#image=tf.keras.layers.Input(shape=(256,256,2))
logits=network(images)
print(logits.shape)

            
            
    

            
            
        
        
        
        
