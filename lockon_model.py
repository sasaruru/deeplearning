import tensorflow as tf

def inference_deep(images_placeholder, keep_prob, image_size, num_classes):

    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_valiable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_valiable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #W_conv3 = weight_variable([3, 3, 64, 128])
    #b_conv3 = bias_valiable([128])
    #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    w = int(image_size / 4)
    W_fc1 = weight_variable([w*w*64, 1024])
    b_fc1 = bias_valiable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, w*w*64])

    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1  
    h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)

    W_fc2 = weight_variable([1024,num_classes])
    b_fc2 = bias_valiable([num_classes])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_conv=tf.nn.softmax(h_fc2)

    return y_conv

def weight_variable(shape):
    initiail = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initiail)

def bias_valiable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def loss(logits, labels):
    #http://qiita.com/ikki8412/items/3846697668fc37e3b7e0
    #cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
