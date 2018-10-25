import tensorflow as tf
import numpy as np
import module.config as config
import module.adversial as adversial

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate


def depth_loss(pred, truth, mask):
    """
    pred=nx12xhxwx1
    truth="
    mask="
    return normalized loss scalar
    """
    loss = tf.subtract(pred, truth)
    loss = tf.abs(loss)
    loss = tf.multiply(loss, mask)
    nloss = tf.reduce_mean(loss)
    # nloss=nloss*pred.get_shape()[0].value
    return nloss


def normal_loss(pred, truth, mask):
    """
    pred=nx12xhxwx1
    truth="
    mask="
    return normalized loss scalar
    """
    loss = tf.subtract(pred, truth)
    m = mask
    mask = tf.stack((mask, m, m), -1)
    loss = tf.abs(loss)
    loss = tf.multiply(loss, mask)
    loss = tf.reduce_mean(loss)
    return loss


def mask_loss(pred, truth):
    # [-1,1] -> [0,1]
    pred = pred * 0.5 + 0.5
    truth = truth * 0.5 + 0.5

    loss = tf.multiply(truth, tf.log(tf.maximum(1e-6, pred)))
    loss = loss + tf.multiply((1 - truth), tf.log(tf.maximum(1e-6, 1 - pred)))
    loss = tf.reduce_sum(-loss)
    # nloss=loss/np.prod(truth.get_shape().as_list()[1:])
    nloss = loss / (256 * 256)
    return nloss

def total_loss(pred, truth):
    """
    pred=n,12,h,w,5
    truth is a tuple
    
    returns total pixel loss
    """

    truth = truth[0]
    truth=tf.reshape(truth,[-1,12,256,256,5])
    depth_pred = pred[:, :, :, :, 0]
    depth_truth = truth[:,:, :, :, 0]
    normal_pred = pred[:, :, :, :, 1:4]
    normal_truth = truth[:,:, :, :, 1:4]
    mask_pred = pred[:, :, :, :, 4]
    mask_truth = truth[:,:, :, :, 4]

    dl = depth_loss(depth_pred, depth_truth, mask_truth)
    nl = normal_loss(normal_pred, normal_truth, mask_truth)
    ml = mask_loss(mask_pred, mask_truth)
    return (dl + ml + nl)
def get_adversial_loss(pred,truth):
    """
    pred :n,12,256,256,5
    returns loss_gen,loss_adv
    """
    total_pixel_loss=total_loss(pred,truth)
    #finding probabilities
    view_pred=tf.transpose(pred,[1,0,2,3,4])
    view_truth=tf.transpose(truth[0],[1,0,2,3,4])# so that views are in the first dimension
    view_truth=tf.reshape(view_truth,[12,-1,256,256,5])
    #[12,?,256,256,5]
    
    prob_pred=adversial.discriminate(view_pred[0,:,:,:,:])
    prob_truth=adversial.discriminate(view_truth[0,:,:,:,:])
    
    for i in range(1,12):
        temp_pred=adversial.discriminate(view_pred[i,:,:,:,:])
        temp_truth=adversial.discriminate(view_truth[i,:,:,:,:])
        prob_pred=tf.concat([prob_pred,temp_pred],axis=0)
        prob_truth=tf.concat([prob_truth,temp_truth],axis=0)
        
    #adversory loss
    #prob_pred: atensor of dim 1 with probs joines at the tail
    loss_on_truth = tf.reduce_sum(-tf.log(tf.maximum(prob_truth, 1e-6)))
    loss_on_pred = tf.reduce_sum(-tf.log(tf.maximum(1.0-prob_pred, 1e-6)))#pred are of class 0
    loss_adv=loss_on_truth+loss_on_pred
    #generators loss
    
    loss_gen_adv=tf.reduce_sum(-tf.log(tf.maximum(prob_pred, 1e-6)))
    #for the adversory prediction should be of class 1 ,same as truth
    loss_gen=config.lambda_pixel*total_pixel_loss+config.lambda_adv*loss_gen_adv
    
    return loss_gen,loss_adv

#####
import numpy as np
def test():
    pred=np.random.randn(4,12,256,256,5)*255
    pred=pred.astype(np.float32)
    truth=np.random.randn(4,12,256,256,5)*255
    truth=truth.astype(np.float32)
    truth=tuple([truth])
    lg,la=get_adversial_loss(pred,truth)
    print(lg)
    print(la)
if __name__=="__main__":
    test()
    
    

    
    
