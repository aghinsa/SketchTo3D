import tensorflow as tf
import numpy as np
import os

import config
config.init()

import model
import loss
import data

main_dir=config.main_dir
training_iter=config.training_iter
batch_size=config.batch_size
learning_rate=config.learning_rate
name_list_path=config.name_list_path

name_list=data.file_to_list(name_list_path)
source_iterator,target_iterator=data.load_data(name_list)
source=source_iterator.get_next()
target=target_iterator.get_next()

pred=model.encoderNdecoder(source)
cost=loss.total_loss(pred,target)
accuracy,_=tf.metrics.accuracy(labels=target_next,predictions=pred)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    for epoch in range(training_iter):
        print("Starting epoch {}".format(epoch+1))
        sess.run(source_iterator.initializer)
        sess.run(target_iterator.initializer)

        for batch in range((name_list.shape[0]//batch_size)+1):
            print("training batch {} .....".format(batch+1))
            opt=sess.run(optimizer)
            l=sess.run(cost)
            acc=sess.run(accuracy)
        print("loss = {} \n accuracy = {} ".format(l,acc))
