def apply_mask(content, mask):
	m=mask;
	channel=content.shape[-1];
	for i in range(channel):
		m=tf.concat([m,m],axis=-1)
	print(m.shape)
	masked=tf.multiply(content,m)
	print(masked.shape)
	return masked


def apply_mask(content, mask):
    m=tf.reshape(mask,[-1,256,256,1])
    try:
        channel = content.get_shape()[-1].value
    except:
        channel = content.shape[-1]
    print(content.shape)
    #content=tf.reshape(content,[-1,256,256,channel])
    if channel > 1:
        m = tf.tile(mask, [1,1,channel])
    masked=tf.where(tf.greater(m, 0.0), content, tf.ones_like(content))
    print("in apply_mask")
    print(masked.shape)
    masked=tf.reshape(masked,[256,256,channel])
    return masked
