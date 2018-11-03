





import tensorflow as tf
import numpy as np
import os
import time
import sys

import module.data as data
import module.loss as loss
import module.model as model
import module.adversial as adversial
import module.config as config
from module.memory_saving_gradients import gradients

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate
name_list_path = config.name_list_path


if(config.is_training):
    with tf.name_scope("Data_Loading"):
        name_list = data.file_to_list(name_list_path)
        source_iterator, target_iterator = data.load_data(name_list)
        source = source_iterator.get_next()
        target = target_iterator.get_next()

    #predictions
    pred = model.encoderNdecoder(source)
    #accuracy
    accuracy=tf.abs(pred-target[0])
    num_pixels=tf.constant(12*256*256*5,dtype=tf.float32)
    accuracy=tf.reduce_sum(accuracy)/num_pixels
    # Global Step
    global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

    # Gather Losses
    with tf.name_scope("Total_Losses"):
        total_pixel_loss=loss.total_loss(pred,target)


    if(config.is_adversial):
        #probabilities
        view_pred=tf.transpose(pred,[1,0,2,3,4])
        view_truth=tf.transpose(target[0],[1,0,2,3,4])# so that views are in the first dimension
        view_truth=tf.reshape(view_truth,[12,-1,256,256,5])
        #[12,?,256,256,5]
        split_preds=tf.split(view_pred,12,axis=0)
        stacked_pred=[tf.squeeze(x,axis=0) for x in split_preds]
        stacked_pred=tf.concat(stacked_pred,axis=0)
        #[?,256,256,5]
        split_truths=tf.split(view_truth,12,axis=0)
        stacked_truth=[tf.squeeze(x,axis=0) for x in split_truths]
        stacked_truth=tf.concat(stacked_truth,axis=0)
        #[?,256,256,5]
        probs_input=tf.concat([stacked_pred,stacked_truth],axis=0)
        probs=adversial.discriminate(probs_input)
        prob_pred,prob_truth=tf.split(probs,2,axis=0)

        loss_gen,loss_adv=loss.get_adversial_loss(prob_pred,prob_truth,total_pixel_loss)



    if(config.is_adversial):
        print("Using adversarial network")
        all_variables=tf.trainable_variables()
        generator_vars=[var for var in all_variables if 'coder' in var.name]
        disc_vars=[var for var in all_variables if 'discri' in var.name]
        
        # Encoder decoder
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads=gradients(loss_gen,generator_vars, checkpoints= "memory")
        grads_and_vars=list(zip(grads,generator_vars))
        #optimizer1 = optimizer1.apply_gradients(grads_and_vars,global_step=global_step)
        optimizer1 = optimizer1.apply_gradients(grads_and_vars)

        # Discriminator
        grads2=gradients(loss_adv,disc_vars, checkpoints= "memory")
        grads_and_vars2 =list(zip(grads2,disc_vars))
        optimizer2=tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer2 = optimizer2.apply_gradients(grads_and_vars2,global_step=global_step) 
        optimizer2 = optimizer2.apply_gradients(grads_and_vars2) 
        
    else:
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_pixel_loss,global_step=global_step)

    init = tf.global_variables_initializer()
    linit = tf.local_variables_initializer()

    n_batches=name_list.shape[0]//batch_size

    

    #summary
    with tf.name_scope("summaries"):
        #tf.summary.scalar('Pixel loss',total_pixel_loss)
        total_loss_summary=tf.summary.scalar('Gen (Total) loss',loss_gen)
        adv_loss_summary=tf.summary.scalar('Adv loss',loss_adv)
        #tf.summary.scalar('Accuracy',accuracy)

        # image summaries
        is1=tf.summary.image("Input",tf.expand_dims(tf.reshape(source[0],[-1,256,256,2])[:,:,:,0],axis=-1), max_outputs=4)
        target_display=tf.concat([stacked_truth[:,:,:,1:4],tf.expand_dims(stacked_truth[:,:,:,0],axis=-1)],axis=-1)
        is2=tf.summary.image("Ground Truth",target_display,max_outputs=4)
        pred_display=tf.concat([stacked_pred[:,:,:,1:4],tf.expand_dims(stacked_pred[:,:,:,0],axis=-1)],axis=-1)
        is3=tf.summary.image("Ground Prediction",pred_display,max_outputs=4)
        perfomance_summary=tf.summary.merge([total_loss_summary,adv_loss_summary])
        image_summary=tf.summary.merge([is1,is2,is3])

    # Checkpoints

    print('Training...')

    ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
#with tf.Session() as sess:
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth=True
sess = tf.Session(config=sess_config)
sess.run(init)
sess.run(linit)
if(config.is_training):
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5, max_to_keep=2)
        # saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoints_dir)) # search for checkpoint file
        graph = tf.get_default_graph()
    else:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5, 
                            max_to_keep=2)
        global_step = 0
        
    print("Global Step:{}".format(global_step))
    sess.run(source_iterator.initializer)
    sess.run(target_iterator.initializer)
    
    
    tf.summary.FileWriterCache.clear()
    train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
    eval_writer = tf.summary.FileWriter(config.eval_log_dir, sess.graph)
    
    for epoch in range(training_iter):
        tic = time.clock()
        print("Starting epoch {}".format(epoch + 1))
        sess.run(source_iterator.initializer)
        sess.run(target_iterator.initializer)
        batch=1
        
        while(True):
            try:
                global_step=global_step+1
                #print('Step : {}'.format(global_step))
                print("\t {}% completed ..".format((batch)*200*config.batch_size/n_batches),end=' ')
                if(config.is_adversial):
                    opt1 = sess.run(optimizer1)
                    opt2=sess.run(optimizer2)
                    l=sess.run(loss_gen)
                else:
                    opt1 = sess.run(optimizer1)
                    l=sess.run(total_pixel_loss)
                print("{}.Total loss : {}".format(epoch+1,l))
                
                
                #writing image (eval)summaries
                batch=batch+1
                
                p_summary=sess.run(perfomance_summary)
                train_writer.add_summary(p_summary, global_step)
                if((batch)%200==0):
                    i_summary=sess.run(image_summary)
                    train_writer.add_summary(i_summary, global_step)
                if((global_step)%500==0):
                    saver.save(sess, os.path.join(config.checkpoints_dir,'model.ckpt'), 
                                global_step=global_step)
                
                
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(config.checkpoints_dir,'model.ckpt'), 
                            global_step=global_step)
                print()
                print("Total batches : {}".format(batch))
                print("\t Epoch {} summary".format(epoch + 1))
                print("\t loss = {} ".format(l))
                toc = time.clock()
                print("\t Time taken :{}".format((toc - tic) / 60))
                break
                
    #if(not config.is_training):
