import tensorflow as tf
import numpy as np
import config

main_dir=config.main_dir
training_iter=config.training_iter
batch_size=config.batch_size
learning_rate=config.learning_rate

def depth_loss(pred,truth,mask):
    """
    pred=nx12xhxwx1
    truth="
    mask="
    return normalized loss scalar
    """
    loss=tf.subtract(pred,truth)
    loss=tf.abs(loss)
    loss=tf.multiply(loss,mask)
    nloss=tf.reduce_mean(loss)
    #nloss=nloss*pred.get_shape()[0].value
    return nloss

def normal_loss(pred,truth,mask):
    """
    pred=nx12xhxwx1
    truth="
    mask="
    return normalized loss scalar
    """
    nloss=depth_loss(pred,truth,mask)
    #nloss=loss*pred.get_shape()[3]
    return nloss

def mask_loss(pred,truth):
    #[-1,1] -> [0,1]
    pred=pred*0.5+0.5
    truth=truth*0.5+0.5

    loss=tf.multiply(truth,tf.log(tf.maximum(1e-6,pred)))
    loss=loss+tf.multiply((1-truth),tf.log(tf.maximum(1e-6,1-pred)))
    loss=tf.reduce_sum(-loss)
    #nloss=loss/np.prod(truth.get_shape().as_list()[1:])
    nloss=loss/(256*256)
    return nloss

def total_loss(pred,truth):
    """
    pred=nxhxwx5
    truth is a tuple
    """

    truth=truth[0]
    depth_pred=pred[:,:,:,:,0]
    depth_truth=truth[:,:,:,:,0]
    normal_pred=pred[:,:,:,:,1:4]
    normal_truth=truth[:,:,:,:,1:4]
    mask_pred=pred[:,:,:,:,4]
    mask_truth=truth[:,:,:,:,4]

    dl=depth_loss(depth_pred,depth_truth,mask_truth)
    nl=normal_loss(normal_pred,normal_truth,mask_truth)
    ml=mask_loss(mask_pred,mask_truth)
    return (dl+ml+nl)
