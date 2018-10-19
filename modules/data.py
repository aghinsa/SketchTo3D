
import tensorflow as tf
import numpy as np
import module.config as config
import os
import cv2

main_dir = config.main_dir
sketch_dir = config.sketch_dir
dnfs_dir = config.dnfs_dir
batch_size = config.batch_size
training_iter = config.training_iter

def file_to_list(text):
    """
    return ndarray
    """
    f = open(text, 'r')
    text_list = []
    for line in f:
        text_list.append(line[0:-1])
    f.close()
    print("No: of files : {}".format(len(text_list)))
    return np.asarray(text_list)


def normalize_image(image):
        # normalize to [-1.0, 1.0]
    if image.dtype == np.uint8:
        return image.astype("float") / 127.5 - 1.0
    elif image.dtype == np.uint16:
        return image.astype("float") / 32767.5 - 1.0
    else:
        return image.astype("float")


def read_sketch(name, value=0):
    # value 0 to 3
    """
    name:passed from name_list
    value:[0..3] each gives a dataset of [256,256,2]
    
    return:[256,256,2] dtype=float32
    """
    name = name.decode('utf-8')

    source_directory = os.path.join(sketch_dir, name)
    f_dir = os.path.join(source_directory, 'sketch-F-{}.png'.format(value))

    c0 = cv2.imread(f_dir, 0)
    c0 = normalize_image(c0)
    s_dir = os.path.join(source_directory, 'sketch-S-{}.png'.format(value))
    c1 = cv2.imread(s_dir, 0)
    c1 = normalize_image(c1)
    temp = np.stack((c0, c1), axis=-1)
    return np.float32(temp)


def source(name_list,eager=False):
    """
    return :Source dataset iterator
        if not eager:
        initialiazable dataset iterator 
        
        if eager:
            one_shot_iterator
            
    Note:retrun of iter.get_next is  a tuple
    """
    source_dataset = tf.data.Dataset.from_tensor_slices(name_list)
    
    def map_fn(name): return tf.py_func(read_sketch, [name], [tf.float32])

    if not eager:
        source_dataset=source_dataset.apply(tf.contrib.data.map_and_batch(
            lambda name:tf.py_func(read_sketch,[name],[tf.float32]),
            batch_size,
            num_parallel_batches=config.num_parallel_batches))
            
    else:
        source_dataset=source_dataset.map(lambda name:tf.py_func(read_sketch,[name],[tf.float32]))
        
    source_dataset = source_dataset.prefetch(
        buffer_size=config.prefetch_buffer_size)
        
    if not eager:
        iter = source_dataset.make_initializable_iterator()
    else:
        iter = source_dataset.make_one_shot_iterator()
    return iter


def read_dnfs(name, views=12):
    """
    name:passed from name_list
    return:[12,256,256,5] dtype=float32
    """
    name = name.decode('utf-8')
    target_directory = os.path.join(dnfs_dir, name)
    va = []  # view array
    for count in range(views):
        encoded = cv2.imread(
            os.path.join(
                target_directory,
                "dn-256-{}.png".format(count)),
            cv2.IMREAD_UNCHANGED)
        mask = encoded[:, :, 0] > 0.9
        depth_map = normalize_image(encoded[:, :, 0])
        nx = normalize_image(encoded[:, :, 1])
        ny = normalize_image(encoded[:, :, 2])
        nz = normalize_image(encoded[:, :, 3])
        temp = np.stack((depth_map, nx, ny, nz, mask), axis=-1)
        va.append(temp)
    results = np.stack(
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
        axis=0)
    print("reading data of {} ".format(name))
    return np.float32(results)


def target(name_list, eager = False):
    """
    return :target dataset iterator
        if not eager:
        initialiazable dataset iterator 
        
        if eager:
            one_shot_iterator
            
    Note:retrun of iter.get_next is  a tuple
    """
    target_dataset = tf.data.Dataset.from_tensor_slices(name_list)
    if not eager:
       target_dataset=target_dataset.apply(tf.contrib.data.map_and_batch(
           lambda name:tf.py_func(read_dnfs,[name],[tf.float32]),
           batch_size,
           num_parallel_batches=config.num_parallel_batches))
    else:
       target_dataset=target_dataset.map(lambda name:tf.py_func(read_sketch,[name],[tf.float32]))
       
    terget_dataset = target_dataset.prefetch(
        buffer_size=config.prefetch_buffer_size)

    if not eager:   
        iter = target_dataset.make_initializable_iterator()
    else:
        iter = target_dataset.make_one_shot_iterator()
    return iter


def load_data(name_list, eager = False):
    """
    return iterators
    """
    si = source(name_list, eager = eager)
    ti = target(name_list, eager = eager)
    return si, ti
