import os
import cv2
import numpy as np
import tensorflow as tf
import module.model as model

def unnormalize_image(image, maxval=255.0):
	# restore image to [0.0, maxval]
	return (image+1.0)*maxval*0.5

def saturate_image(image, dtype=tf.uint8):
	return tf.saturate_cast(image, dtype)

def encode_batch_images(batch):
	"""
		input:
			batch:  n x H x W x C   input images batch
		output:
			packed: n x String      output PNG-encoded strings
	"""
	# output:
	unpacked = tf.unstack(batch)
	num = len(unpacked)
	encoded = [None] * num
	for k in range(num):
		encoded[k] = tf.image.encode_png(unpacked[k])
	return tf.stack(encoded)    

def read_input(source_directory, value=0):
    """
    return:[256,256,2] dtype=float32
    """
    f_dir = os.path.join(source_directory, 'sketch-F-{}.png'.format(value))
    c0 = cv2.imread(f_dir, 0)
    c0=cv2.resize(c0,(256,256))
    c0 = normalize_image(c0)
    s_dir = os.path.join(source_directory, 'sketch-S-{}.png'.format(value))
    c1 = cv2.imread(s_dir, 0)
    c1=cv2.resize(c1,(256,256))
    c1 = normalize_image(c1)
    temp = np.stack((c0, c1), axis=-1)
    return np.float32(temp)
	
def normalize_image(image):
        # normalize to [-1.0, 1.0]
    if image.dtype == np.uint8:
        return image.astype("float") / 127.5 - 1.0
    elif image.dtype == np.uint16:
        return image.astype("float") / 32767.5 - 1.0
    else:
        return image.astype("float")
		
def write_image(name, image):
	"""
		input:
			name:  String     file name
			image: String     PNG-encoded string
	"""
	path = os.path.dirname(name)
	if not os.path.exists(path):
		os.makedirs(path)
	file = open(name, 'wb')
	file.write(image)
	file.close()
    
def collect(main_dir,sess):
    input_sketch=os.path.join(main_dir,'hires')
    output_dir=os.path.join(main_dir,'output')
    output_maps=os.path.join(output_dir,'images')
    output_results=os.path.join(output_dir,'result')
    output_prefix = 'dn14'
    
    input_image=read_input(input_sketch)
    
    preds=model.encoderNdecoder(input_image)
	# #write input image
    # img_input = saturate_image(unnormalize_image(input_image, maxval=65535.0), dtype=tf.uint16)
    # png_input = encode_batch_images(img_input)
    # name_input = os.path.join(output_maps,'input.png')
    # write_image(name_input,png_input)
	# 
    # for view in range(12):
    #     preds_depth=preds[0,view,:,:,0]
    #     preds_normal=preds[0,view,:,:,1:4]
    #     preds_mask=preds[0,view,:,:,4]
    #     #result
    #     img_output=saturate_image(unnormalize_image(preds[0:view,:,:,:],maxval=65535.0),dtype=tf.uint16)
    #     png_output=encode_batch_images(img_output)
    #     name_output = os.path.join(output_maps,('pred-'+output_prefix+'--'+view+'.png'))
    #     write_image(name_output,png_output)
    #     #normals
    #     name_normal = os.path.join(output_maps,('normal-'+output_prefix+'--'+view+'.png'))
    #     img_normal = saturate_image(unnormalize_image(preds_normal,
	# 						maxval=65535.0), dtype=tf.uint16)
    #     png_normal = encode_batch_images(img_normal)
    #     write_image(name_normal,png_normal)
    #     #depth
    #     name_depth = os.path.join(output_maps,('depth-'+output_prefix+'--'+view+'.png'))
    #     img_depth = saturate_image(unnormalize_image(preds_depth, 
	# 							maxval=65535.0), dtype=tf.uint16)
    #     png_depth = encode_batch_images(img_depth)
    #     write_image(name_depth,png_depth)
    #     #mask
    #     name_mask = os.path.join(output_maps,('mask-'+output_prefix+'--'+view+'.png'))
    #     img_mask = saturate_image(unnormalize_image(preds_mask, 
	# 							maxval=65535.0), dtype=tf.uint16)
    #     png_mask = encode_batch_images(img_mask)
    #     write_image(name_mask,png_mask)
    #     #Export to results
    #     img_output=saturate_image(unnormalize_image(preds[0:view,:,:,:],
	# 	 								maxval=65535.0), dtype=tf.uint16)
    #     png_output=encode_batch_images(img_output)
    #     name_output = os.path.join(output_results,('pred-'+output_prefix+'--'+view+'.png'))
    #     write_image(name_output,png_output)

check_dir=
#img_dir/hires/
img_dir=
sess=tf.Session()
saver = tf.train.import_meta_graph(os.path.join(check_dir,'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('check_dir'))
pred=collect(img_dir,sess)
#prediction = sess.run(y4,feed_dict={x:sampletest})

