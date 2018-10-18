import os
import tensorflow as tf
import numpy as np
import cv2
import config

main_dir=config.main_dir
training_iter=config.training_iter
batch_size=config.batch_size
learning_rate=config.learning_rate

def file_to_list(text):
    f=open(text,'r')
    text_list=[]
    for line in f:
        text_list.append(line[0:-1])
    f.close()
    return text_list

def normalize_image(image):
	# normalize to [-1.0, 1.0]
	if image.dtype == np.uint8:
		return image.astype("float")/127.5-1.0
	elif image.dtype == np.uint16:
		return image.astype("float")/32767.5-1.0
	else:
		return image.astype("float")
def read_sketch(source_directory,value,size=256):
    #value 0 to 3
    """
    direcorty:main/sketch/
    view:f s
    return: size x size x 2
    """
    # temp=np.empty((size,size))
    f_dir=os.path.join(source_directory,'sketch-F-{}.png'.format(value))
    print(f_dir)
    c0=cv2.imread(f_dir,0)
    print(c0.dtype)
    c0=normalize_image(c0)
    s_dir=os.path.join(source_directory,'sketch-S-{}.png'.format(value))
    c1=cv2.imread(s_dir,0)
    c1=normalize_image(c1)
    temp=np.stack((c0,c1),axis=-1)
    return temp

def read_dnfs(target_directory,views=12,size=256):
    """
    direcorty:main/dnfs/subject
    view:0 .. 11
    return size x size x 5
    """
    va=[]#view array
    for count in range(views):
        print("view{}....".format(count+1))
        print("reading dnfs: {}".format(target_directory))
        encoded=cv2.imread(os.path.join(target_directory,"dn-256-{}.png".format(count)),cv2.IMREAD_UNCHANGED)
        print("decoding.........")
        mask=encoded[:,:,0]>0.9
        depth_map=normalize_image(encoded[:,:,0])
        print("depth done...........")
        nx=normalize_image(encoded[:,:,1])
        ny=normalize_image(encoded[:,:,2])
        print("x,y done.........")
        nz=normalize_image(encoded[:,:,3])
        temp=np.stack((depth_map,nx,ny,nz,mask),axis=-1)
        va.append(temp)
        print("stacking")
    results=np.stack((va[0],va[1],va[2],va[3],va[4],va[5],va[6],va[7],va[8],va[9],va[10],va[11]),axis=0)
    print("stack shape..........")
    print(results.shape)
    return results

def get_subject_source(name,value=0):
    #main dir is global
    sketch=os.path.join(main_dir,'sketch',name)
    source=read_sketch(sketch,value)
    return source
def get_subject_target(name):
    dnfs=os.path.join(main_dir,'dnfs',name)
    target=read_dnfs(dnfs)
    return target
