# BATCH TRAINING DATASET api_docs

si, ti = da.load_data(name_list, batched=True)
sn = si.get_next()
tn = ti.get_next()

with tf.Session() as sess:
    for epoch in range(training_iter):
        print("Starting epoch {}".format(epoch + 1))
        sess.run(si.initializer)
        sess.run(ti.initializer)
        while(1):
            try:
                a = sess.run(sn)
            except BaseException:
                pass
            try:
                b = sess.run(tn)
            except tf.errors.OutOfRangeError:
                break
