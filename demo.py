import tensorflow as tf
import numpy as np
from Data import Data
import cv2
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers import batch_norm

def batch_normalization_layer(input,name):
    axis = list(range(len(input.get_shape())-1))
    mean,variance = tf.nn.moments(input,axis)
    beta = tf.get_variable('beta'+name,initializer=tf.zeros_initializer,shape=input.get_shape()[-1],dtype=tf.float32)
    gamma = tf.get_variable('gamma'+name,initializer=tf.ones_initializer,shape=input.get_shape()[-1],dtype=tf.float32)

    moving_mean = tf.get_variable('moving_mean'+name,initializer=tf.zeros_initializer,
                                  shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    moving_variance = tf.get_variable('moving_var'+name,initializer=tf.zeros_initializer,
                                      shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,0.999)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,0.999)

    tf.add_to_collection('mean'+name,update_moving_mean)
    tf.add_to_collection('variance'+name,update_moving_variance)

    return tf.nn.batch_normalization(input,mean,variance,beta,gamma,0.001)

y = tf.placeholder(tf.float32, [64, 32, 128, 3])
x = tf.placeholder(tf.float32, [64, 32, 128, 3])
h = tf.layers.conv2d(x,64,[5,5],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.layers.max_pooling2d(h,[2,2],[2,2])


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.layers.max_pooling2d(h,[2,2],[2,2])


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.layers.max_pooling2d(h,[2,2],[2,2])


h = tf.layers.conv2d(h,64,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.layers.max_pooling2d(h,[2,2],[2,2])


h = tf.layers.conv2d(h,64,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.image.resize_nearest_neighbor(h,[4,16])


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.image.resize_nearest_neighbor(h,[8,32])


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.image.resize_nearest_neighbor(h,[16,64])


h = tf.layers.conv2d(h,64,[5,5],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.image.resize_nearest_neighbor(h,[32,128])


h = tf.layers.conv2d(h,3,[1,1],[1,1],padding='SAME')
out = batch_norm(h)
out = tf.nn.relu(out)

stackcost = tf.reduce_mean(tf.square(tf.subtract(y, out)))
opt = tf.train.AdamOptimizer(0.85).minimize(stackcost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

data = Data('../data/data.npz', 64)
for epoch in range(20000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        # input = sess.run(h,feed_dict={x:})
        gaussianNoise = 0.01*np.random.normal(size=[64,12288]).reshape([64,32,128,3])

        traindata = data.batch_size([-1, 32, 128, 3])
        _, cost = sess.run((opt, stackcost), feed_dict={x: traindata+gaussianNoise,y:traindata})

        avg_cost += cost / len(data.train_data) * 64
        if i % 50 == 0 and epoch %50==0:
            rebuildimage = sess.run(out, feed_dict={x: traindata})
            cv2.imwrite('./outputImage/input.hdr', traindata[0][:, :, ::-1])
            # print(output.shape)
            cv2.imwrite('./outputImage/' + str(epoch) + '_output.hdr',
                        np.reshape(rebuildimage[0], [32, 128, 3])[:, :, ::-1])


    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
