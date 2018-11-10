import module.data as data
import module.config as config
import tensorflow as tf
main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate
name_list_path = config.name_list_path

name_list = data.file_to_list(name_list_path)
source_iterator, target_iterator = data.load_data(name_list)
source = source_iterator.get_next()
target = target_iterator.get_next()

with tf.Session() as sess:
    sess.run(source_iterator.initializer)
    sess.run(target_iterator.initializer)
    print(sess.run(source[0]).shape)
    print("target")
    print(sess.run(target[0]).shape)
