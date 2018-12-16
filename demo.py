import tensorflow as tf
import numpy as np
from Data import Data
import cv2
from tensorflow.python.training import moving_averages
def batch_normalization_layer(input):
    axis = list(range(len(input.get_shape())-1))
    mean,variance = tf.nn.moments(input,axis)
    beta = tf.get_variable('beta',initializer=tf.zeros_initializer,shape=input.get_shape()[-1],dtype=tf.float32)
    gamma = tf.get_variable('gamma',initializer=tf.ones_initializer,shape=input.get_shape()[-1],dtype=tf.float32)

    moving_mean = tf.get_variable('moving_mean',initializer=tf.zeros_initializer,
                                  shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    moving_variance = tf.get_variable('moving_var',initializer=tf.zeros_initializer,
                                      shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,0.999)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,0.999)

    tf.add_to_collection('mean',update_moving_mean)
    tf.add_to_collection('variance',update_moving_variance)

    return tf.nn.batch_normalization(input,mean,variance,beta,gamma,0.001)

x = tf.placeholder(tf.float32, [64, 32, 128, 3])
h = tf.nn.conv2d(x,tf.get_variable('w1',[5,5,3,64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False)),
                 [1,1,1,1],padding='SAME')+tf.get_variable('b1',[64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False))
h = batch_normalization_layer(h)
h = tf.nn.relu(h)
# h = tf.nn.relu(
#
#     tf.nn.conv2d(h,tf.get_variable('w3',[3,3,64,96],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False)),
#                  [1,1,1,1],padding='SAME')+tf.get_variable('b3',[96],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False))
#     )


# h = tf.nn.dropout(h,0.75)
# h = tf.layers.conv2d_transpose(h,64,[5,5],[2,2],activation=tf.nn.relu,
#                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                padding='SAME'
#                                )
# h = tf.nn.relu(
# batch_norm(
#     tf.nn.conv2d_transpose(
#         h,tf.get_variable('w4',[5,5,64,64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False)),[64,16,64,64],[1,1,1,1],
#         padding='SAME'
#     )+tf.get_variable('b4',[64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False))
# )
# )
# h = tf.nn.relu(
# batch_norm(
#     tf.nn.conv2d_transpose(
#         h,tf.get_variable('w5',[5,5,64,64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False)),[64,32,128,64],[1,2,2,1],
#         padding='SAME'
#     )+tf.get_variable('b5',[64],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False))
# )
# )
# h = tf.nn.dropout(h,0.75)
#h = tf.nn.relu(batch_norm(h))
out = tf.nn.relu(
    tf.nn.conv2d(h,tf.get_variable('w2',[1,1,64,3],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False)),
                 [1,1,1,1],padding='SAME')+tf.get_variable('b2',[3],tf.float32,tf.contrib.layers.xavier_initializer(uniform=False))
                            )
stackcost = tf.reduce_mean(tf.square(tf.subtract(x, out)))
opt = tf.train.AdamOptimizer(0.0085).minimize(stackcost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

data = Data('../data/data.npz', 64)
for epoch in range(20000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        # input = sess.run(h,feed_dict={x:})
        traindata = data.batch_size([-1, 32, 128, 3])
        _, cost = sess.run((opt, stackcost), feed_dict={x: traindata})
        avg_cost += cost / len(data.train_data) * 64
        if i % 50 == 0 and epoch %50==0:
            rebuildimage = sess.run(out, feed_dict={x: traindata})
            cv2.imwrite('./outputImage/input.hdr', traindata[0][:, :, ::-1])
            # print(output.shape)
            cv2.imwrite('./outputImage/' + str(epoch) + '_output.hdr',
                        np.reshape(rebuildimage[0], [32, 128, 3])[:, :, ::-1])


    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
