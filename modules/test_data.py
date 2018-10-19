import tensorflow as tf
tf.enable_eager_execution()

import data
import loss
import model
import numpy as np
import os
import time

import module.config as config

tfe= tf.contrib.eager

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate
name_list_path = config.name_list_path

name_list = data.file_to_list(name_list_path)
source_iterator, target_iterator = data.load_data(name_list, eager = True)

model = model.keras_model()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
      cost = loss.total_loss(pred, target)
  return tape.gradient(cost, model.variables), cost

def train_op(source,target):
    pred = model(source)
    
    accuracy, _ = tfe.metrics.accuracy(labels=target, predictions=pred)
    # Calculate derivatives of the input function with respect to its parameters.
    grads, loss = grad(model, x, y)
    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads, model.variables),
                                global_step=tf.train.get_or_create_global_step())

    return loss, accuracy
                                
# for i,image in enumerate(source_iterator):
#     print(i)
#     print(image[0])
#     print(image[0].shape)

for epoch in range(training_iter):
        tic = time.clock()
        print("Starting epoch {}".format(epoch + 1))

        l, acc = 0,0
        for i, (source,target) in enumerate(zip(source_iterator, target_iterator)):
            print("training batch {} .....".format(batch + 1))
            l, acc = train_op(source,target)
        print()
        print("Epoch {} summary".format(epoch + 1))
        print("     loss = {} ".format(l))
        print("     accuracy = {}".format(acc))
        toc = time.clock()
        print("     Time taken :{}".format((toc - tic) / 60))

