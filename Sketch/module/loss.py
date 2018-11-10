
import tensorflow as tf
import numpy as np
import module.config as config
import module.adversial as adversial
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as cross_entropy

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate

def depth_loss(pred, truth, mask,normalize):
    """
    pred=nx12xhxwx1
    truth="
    mask="
    return normalized loss scalar
    """
    
    with tf.name_scope("depth_loss"):
        loss = tf.subtract(tf.reshape(pred,[-1,12,256,256]),
                            tf.reshape(truth,[-1,12,256,256]))
        loss = tf.abs(loss)
        loss = tf.multiply(loss, mask)
        if(normalize):
            nloss = tf.reduce_mean(loss)
            nloss=nloss*tf.constant(config.batch_size*12,dtype=tf.float32)
        else:
            nloss=tf.reduce_sum(loss)
    
        return nloss


def normal_loss(pred, truth, mask,normalize):
    """
    pred=nx12xhxwx3
    truth="
    mask="
    return normalized loss scalar
    """
    with tf.name_scope("normal_loss"):
        loss = tf.subtract(pred, truth)
        m = mask
        mask = tf.stack((mask, m, m), -1)
        loss = tf.square(loss)
        loss = tf.multiply(loss, mask)
        if(normalize):
            nloss = tf.reduce_mean(loss)
            nloss=nloss*tf.constant(config.batch_size*12*3,dtype=tf.float32)
        else:
            nloss=tf.reduce_sum(loss)
        return nloss


def mask_loss(pred, truth,normalize):
    # [-1,1] -> [0,1]
    with tf.name_scope("mask_loss"):
        pred = pred * 0.5 + 0.5
        #truth is already 0,1

        loss = tf.multiply(truth, tf.log(tf.maximum(1e-6, pred)))
        loss = loss + tf.multiply((1 - truth), tf.log(tf.maximum(1e-6, 1 - pred)))
        loss = tf.reduce_sum(-loss)
        if (normalize):
            nloss = loss / tf.constant(12*256 * 256,dtype=tf.float32)
        else:
            nloss=loss
        return nloss

def total_loss(pred, truth,normalize=config.loss_normalize):
    """
    pred=n,12,h,w,5
    truth is a tuple
    
    returns total pixel loss
    """
    with tf.name_scope("total_pixel_loss"):
        truth = truth[0]
        truth=tf.reshape(truth,[-1,12,256,256,5])
        depth_pred = pred[:, :, :, :, 0]
        depth_truth = truth[:,:, :, :, 0]
        normal_pred = pred[:, :, :, :, 1:4]
        normal_truth = truth[:,:, :, :, 1:4]
        mask_pred = pred[:, :, :, :, 4]
        mask_truth = truth[:,:, :, :, 4]
        dl = depth_loss(depth_pred, depth_truth, mask_truth,normalize)
        nl = normal_loss(normal_pred, normal_truth, mask_truth,normalize)
        ml = mask_loss(mask_pred, mask_truth,normalize)
        return (dl + ml + nl)
def get_adversial_loss(prob_pred,prob_truth,total_pixel_loss):
    """
    pred :n,12,256,256,5
    returns loss_gen,loss_adv
    """
    with tf.name_scope("adversarial_loss"):
        #loss_on_truth = tf.reduce_sum(-tf.log(tf.maximum(prob_truth, 1e-6)))
        #loss_on_pred = tf.reduce_sum(-tf.log(tf.maximum(1.0-prob_pred, 1e-6)))#pred are of class 0
        truth_labels=tf.ones(tf.shape(prob_truth)[0],dtype=tf.int32)
        loss_on_truth=cross_entropy(logits=prob_truth,labels=truth_labels)
        loss_on_truth=tf.reduce_mean(loss_on_truth)
        
        pred_labels=tf.zeros(tf.shape(prob_pred)[0],dtype=tf.int32)
        loss_on_pred=cross_entropy(logits=prob_pred,labels=pred_labels)
        loss_on_pred=tf.reduce_mean(loss_on_pred)
        
        loss_adv=loss_on_truth+loss_on_pred
        #generators loss
        
        #loss_gen_adv=tf.reduce_sum(-tf.log(tf.maximum(prob_pred, 1e-6)))
        pred_labels_adv=tf.ones(tf.shape(prob_pred)[0],dtype=tf.int32)
        loss_gen_adv=cross_entropy(logits=prob_pred,labels=pred_labels_adv)
        loss_gen_adv=tf.reduce_mean(loss_gen_adv)
        #for the adversory prediction should be of class 1 ,same as truth
        loss_gen=(config.lambda_pixel*total_pixel_loss) + (config.lambda_adv*loss_gen_adv)
        
        return loss_gen,loss_adv

    
    

    
    
