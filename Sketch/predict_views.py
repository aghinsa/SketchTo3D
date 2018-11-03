import os
import tensorflow as tf
import cv2
import numpy as np
import module.model as model

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
def test(image,sess,train_dir):
    print("testing")
    saver=tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        try:
            self.step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        except ValueError:
            self.step = 0
    else:
        print('Cannot find any checkpoint file')
        return
    print(ckpt)
def unnormalize_image(image, maxval=255.0):
	# restore image to [0.0, maxval]
	return (image+1.0)*maxval*0.5
def saturate_image(image, dtype=tf.uint8):
	return tf.saturate_cast(image, dtype)
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
def apply_mask(content, mask):
    content=tf.reshape(content,[256,256,-1])
    mask=tf.reshape(mask,[256,256,1])
    channel=content.get_shape().as_list()[-1]
    m=tf.tile(mask,[1,1,channel])
    masked=tf.multiply(content,m)
    return masked

#run this from input image folder
input_image=read_input('.')
input_image=np.reshape(input_image,[-1,256,256,2])
train_dir=('./Checkpoints')
x=tf.placeholder(dtype=tf.float32,shape=[None,256,256,2])
pred=model.encoderNdecoder(x)

main_dir='.'
output_dir=os.path.join(main_dir,'output')
output_image_dir=os.path.join(output_dir,'images')
output_results=os.path.join(output_dir,'result')
output_prefix = 'dn14'


with tf.Session() as sess:
    #writing input image
    in_reshape=tf.reshape(input_image,[256,256,2])
    img_input = saturate_image(unnormalize_image(input_image, maxval=65535.0), dtype=tf.uint16)
    png_input = tf.image.encode_png(img_input[0,:,:,:])
    png_input = sess.run(png_input)
    name_input = os.path.join(output_image_dir,'input.png')
    write_image(name_input, png_input)

    saver = tf.train.Saver()
    #saver.restore(sess,train_dir+'/model.ckpt-8500')
    saver.restore(sess,train_dir+'/model.ckpt-36500')

    feed_dict={x:input_image}
    preds=sess.run(pred,feed_dict)

    ################################################
    preds=preds*255*2
    #preds=apply_mask(preds)
    for view in range(12):
        print(view+1)


        preds_mask=preds[0,view,:,:,4]
        preds_mask=tf.reshape(preds_mask,[256,256,1])
        preds_depth=preds[0,view,:,:,0]
        preds_depth=apply_mask(preds_depth,preds_mask)
        preds_depth=tf.reshape(preds_depth,[256,256,1])
        preds_normal=preds[0,view,:,:,1:4]
        preds_normal=apply_mask(preds_normal,preds_mask)
        #result
        img_output=saturate_image(unnormalize_image(preds[0,view,:,:,0:4],maxval=65535.0),dtype=tf.uint16)
        png_output=tf.image.encode_png(img_output)
        name_output = os.path.join(output_image_dir,('pred-'+output_prefix+'--'+str(view)+'.png'))
        png_output=sess.run(png_output)
        write_image(name_output,png_output)
        #normals
        name_normal = os.path.join(output_image_dir,('normal-'+output_prefix+'--'+str(view)+'.png'))
        img_normal = saturate_image(unnormalize_image(preds_normal,
                            maxval=65535.0), dtype=tf.uint16)
        png_normal = tf.image.encode_png(img_normal)
        png_normal=sess.run(png_normal)
        write_image(name_normal,png_normal)
        #depth
        name_depth = os.path.join(output_image_dir,('depth-'+output_prefix+'--'+str(view)+'.png'))
        img_depth = saturate_image(unnormalize_image(preds_depth,
                                maxval=65535.0), dtype=tf.uint16)
        png_depth = tf.image.encode_png(img_depth)
        png_depth=sess.run(png_depth)
        write_image(name_depth,png_depth)
        #mask
        name_mask = os.path.join(output_image_dir,('mask-'+output_prefix+'--'+str(view)+'.png'))
        img_mask = saturate_image(unnormalize_image(preds_mask,
                                maxval=65535.0), dtype=tf.uint16)
        png_mask = tf.image.encode_png(img_mask)
        png_mask=sess.run(png_mask)
        write_image(name_mask,png_mask)
        #Export to results
        img_output=saturate_image(unnormalize_image(preds[0,view,:,:,1:4],
                                        maxval=65535.0), dtype=tf.uint16)
        png_output=tf.image.encode_png(img_output)
        name_output = os.path.join(output_results,('pred-'+output_prefix+'--'+str(view)+'.png'))
        png_output=sess.run(png_output)
        write_image(name_output,png_output)
