import numpy as np
import tensorflow as tf
from autoencoders.AutoEncoder import Autoencoder_conv2conv, Autoencoder_conv2deconv, Autoencoder_full2deconv
from Data import Data


sess = tf.Session()
saver = tf.train.import_meta_graph('./weights_saver/ae1_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./weights_saver'))

x = tf.placeholder(tf.float32,[64,32,128,3])
h = tf.nn.relu(
    tf.nn.conv2d(
        x,sess.run('ae1_w1:0'),[1,1,1,1],padding='SAME'
    )+sess.run('ae1_b1:0')
)

ae2 = Autoencoder_conv2deconv('ae2',
                              encoder_filter_size=[3,3,64,96],
                              decoder_filter_size=[5,5,64,96],
                              input_shape=[64,32,128,64],
                              optimizer=tf.train.AdamOptimizer(0.00001))



sess.run(tf.global_variables_initializer())



data = Data('../data/data.npz', 64)
for epoch in range(2):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        input = sess.run(h,feed_dict={x:data.batch_size([-1, 32, 128, 3])})
        _, cost = sess.run(ae2.partial_fit(), feed_dict={ae2.x: input})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
