


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

def split_source(data_source):
    """
    in:   -1,4,256,256,2
    return:  -1*4,256,256,2
    a.get_shape()[0]
    """
    batches=tf.split(data_source,config.batch_size,axis=0)
    batches=[tf.squeeze(x) for x in batches]
    batches=tf.concat(batches,axis=0)
    return  batches
    
def multiply_targets(target_source):
    """
    outputs array where eeach input element is stackes four times
    """
    
    batches=tf.split(tf.reshape(target_source,[-1,12,256,256,5]),config.batch_size,axis=0)
    batches=[tf.squeeze(x) for x in batches]
    new_batch=batches[:]
    for i in range(len(batches)):
        for j in range(3):
            new_batch[i]=tf.concat([tf.reshape(new_batch[i],[-1,12,256,256,5]),
                            tf.reshape(batches[i],[-1,12,256,256,5])],axis=0)
    new_batch=tf.concat(new_batch,axis=0)
    # new_batch=tf.split(new_batch,config.batch_size,axis=0)
    # new_batch=[tf.squeeze(x) for x in new_batch]
    # new_batch=tf.concat(new_batch,axis=0)
    return new_batch

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

#img1=cv2.imread(cd)

def read_sketch_aux(name,value=0):
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
    result=np.reshape(temp,[-1,256,256,2])
    
    for value in range(1,4):
        c0 = cv2.imread(f_dir, 0)
        c0 = normalize_image(c0)
        s_dir = os.path.join(source_directory, 'sketch-S-{}.png'.format(value))
        c1 = cv2.imread(s_dir, 0)
        c1 = normalize_image(c1)
        temp = np.stack((c0, c1), axis=-1)
        temp=np.reshape(temp,[-1,256,256,2])
        result=np.concatenate([result,temp],axis=0)
    
    return np.float32(result)

def read_sketch(name,value=0):
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
    result=np.reshape(temp,[-1,256,256,2])
    
    
    return np.float32(result)


def read_dnfs(name):
    """
    name:passed from name_list
    return:[12,256,256,5] dtype=float32
    """
    views=12
    name = name.decode('utf-8')
    target_directory = os.path.join(dnfs_dir, name)
    va = []  # view array
    for count in range(views):
        encoded = cv2.imread(
            os.path.join(
                target_directory,
                "dn-256-{}.png".format(count)),
            cv2.IMREAD_UNCHANGED)
        depth_map = normalize_image(encoded[:, :, 0])
        mask=np.less(depth_map,config.mask_threshold)
        mask=mask.astype(dtype=np.float32)
        nx = normalize_image(encoded[:, :, 1])
        ny = normalize_image(encoded[:, :, 2])
        nz = normalize_image(encoded[:, :, 3])
        
        temp = np.stack((depth_map, nx, ny, nz, mask), axis=-1)
        va.append(temp)
    results = np.stack(
        (va[0],va[1],va[2],va[3],va[4],
        va[5],va[6],va[7],va[8],va[9],va[10],
        va[11]),axis=0)
    return np.float32(results)

# def read_sketch(name):
#     d=read_sketch_aux(name)
#     d=

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
    
    if not eager:
        source_dataset=source_dataset.apply(tf.contrib.data.map_and_batch(
            lambda name:tf.py_func(read_sketch,[name],[tf.float32]),
            batch_size,
            num_parallel_batches=config.num_parallel_batches))
        # source_dataset=source_dataset.map(lambda name:tf.py_func(read_sketch,[name],[tf.float32]))
        # source_dataset=source_dataset.batch(config.batch_size)
    else:
        source_dataset=source_dataset.map(lambda name:tf.py_func(read_sketch,[name,config.sketch_value],[tf.float32]))
        
    source_dataset = source_dataset.prefetch(
                    buffer_size=config.prefetch_buffer_size)
    #source_dataset=source_dataset.repeat(config.training_iter)
        
    if not eager:
        iter = source_dataset.make_initializable_iterator()
    else:
        iter = source_dataset.make_one_shot_iterator()
    return iter



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
       # target_dataset=target_dataset.map(lambda name:tf.py_func(read_dnfs,[name],[tf.float32]))
       # traget_dataset=target_dataset.batch(config.batch_size)
    else:
       target_dataset=target_dataset.map(lambda name:tf.py_func(read_dnfs,[name],[tf.float32]))
       traget_dataset=target_dataset.batch(config.batch_size)
    
    target_dataset = target_dataset.prefetch(
                        buffer_size=config.prefetch_buffer_size)
    #target_dataset=target_dataset.repeat(config.training_iter)

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
