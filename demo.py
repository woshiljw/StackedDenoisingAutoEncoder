import tensorflow as tf
import numpy as np
from Data import Data

x = tf.placeholder(tf.float32, [64, 32, 128, 3])
h = tf.layers.conv2d(x, 64, [5, 5], [1, 1], activation=tf.nn.relu,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                     bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                     padding='SAME'
                     )
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = tf.layers.conv2d(h,96,[3,3],[1,1],activation=tf.nn.relu,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                     bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                     padding='SAME'
                     )
h = tf.layers.conv2d_transpose(h,64,[5,5],[1,1],activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               padding='SAME'
                               )
h = tf.layers.conv2d_transpose(h,64,[5,5],[2,2],activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               padding='SAME'
                               )

out = tf.layers.conv2d(h, 3, [1, 1], [1, 1], activation=tf.nn.relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                       bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                       padding='SAME'
                       )

stackcost = tf.reduce_mean(tf.square(tf.subtract(x, out)))
opt = tf.train.AdamOptimizer().minimize(stackcost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

data = Data('../data/data.npz', 64)
for epoch in range(2000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        # input = sess.run(h,feed_dict={x:})
        _, cost = sess.run((opt, stackcost), feed_dict={x: data.batch_size([-1, 32, 128, 3])})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
