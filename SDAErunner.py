import numpy as np
import tensorflow as tf
from autoencoders.AutoEncoder import Autoencoder_conv2conv,Autoencoder_conv2deconv,Autoencoder_full2deconv
from Data import Data



ae1 = Autoencoder_conv2conv('ae1',
                            encoder_filter_size=[5,5,3,64],
                            dencoder_filter_size=[1,1,64,3],
                            input_shape=[64,32,128,3],
                            optimizer=tf.train.AdamOptimizer(0.0000085))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data = Data('../data/data.npz',64)
for i in range(10):
    avg_cost = 0
    for j in range(50):
        _,cost = sess.run(ae1.partial_fit(),
                          feed_dict={ae1.x:data.train_batch_size([-1,32,128,3])})
        avg_cost+=cost/50
    print(j,": ",avg_cost)
