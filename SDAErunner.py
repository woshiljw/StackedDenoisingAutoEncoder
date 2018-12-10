import numpy as np
import tensorflow as tf
from autoencoders.AutoEncoder import Autoencoder_conv2conv,Autoencoder_conv2deconv,Autoencoder_full2deconv
from Data import Data



ae1 = Autoencoder_conv2conv('ae1',
                            encoder_filter_size=[5,5,3,64],
                            decoder_filter_size=[1,1,64,3],
                            input_shape=[64,32,128,3],
                            optimizer=tf.train.AdamOptimizer(0.0000085))

ae2 = Autoencoder_conv2deconv('ae2',
                              encoder_filter_size=[3,3,64,96],
                              decoder_filter_size=[5,5,64,96],
                              input_shape=[64,32,128,64],
                              optimizer=tf.train.AdamOptimizer(0.00001))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data = Data('../data/data.npz',64)
for epoch in range(20):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        #ae1_output = sess.run(ae1.encode, feed_dict={ae1.x: data.batch_size([-1, 32, 128, 3])})
        _,cost = sess.run(ae1.partial_fit(), feed_dict={ae1.x: data.batch_size([-1, 32, 128, 3])})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
