import numpy as np
import tensorflow as tf
from autoencoders.AutoEncoder import Autoencoder_conv2conv, Autoencoder_conv2deconv, Autoencoder_full2deconv
from Data import Data
from tensorflow.contrib.layers import batch_norm
ae1 = Autoencoder_conv2conv('ae1',
                              [5,5,3,64],
                              [1,1,64,3],
                              [64,32,128,3],
                              tf.train.AdamOptimizer(0.00185))

ae2 = Autoencoder_conv2deconv('ae2',
                              [3,3,64,96],
                              [5,5,64,96],
                              [64,32,128,64],
                              tf.train.AdamOptimizer(0.00085))
ae3 = Autoencoder_conv2deconv('ae3',
                              [3,3,96,96],
                              [3,3,96,96],
                              [64,16,64,96],
                              tf.train.AdamOptimizer(0.00085))

ae4 = Autoencoder_conv2deconv('ae4',
                              [3,3,96,64],
                              [3,3,96,64],
                              [64,8,32,96],
                              tf.train.AdamOptimizer(0.00085))
ae5 = Autoencoder_full2deconv('ae5',
                              tf.train.AdamOptimizer(0.000185))


x = tf.placeholder(tf.float32,[64,32,128,3])
h = tf.nn.relu(tf.nn.conv2d(x,ae1.weight['w1'],[1,1,1,1],padding='SAME')+ae1.weight['b1'])#ae1
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = tf.nn.relu(tf.nn.conv2d(h,ae2.weight['w1'],[1,1,1,1],padding='SAME')+ae2.weight['b1'])#ae2
#h = batch_norm(h)
#h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
#h = tf.nn.relu(tf.nn.conv2d(h,ae3.weight['w1'],[1,1,1,1],padding='SAME')+ae3.weight['b1'])#ae3
#h = batch_norm(h)
#h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
#h = tf.nn.relu(tf.nn.conv2d(h,ae4.weight['w1'],[1,1,1,1],padding='SAME')+ae4.weight['b1'])#ae4
#h = batch_norm(h)
#h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')

#############ae5
#flatten = tf.reshape(h,[64,-1])
#h = tf.nn.relu(tf.matmul(flatten,ae5.weight['w1'])+ae5.weight['b1'])
#h = batch_norm(h)
#h = tf.nn.relu(tf.matmul(h,ae5.weight['w2'])+ae5.weight['b2'])
#h = batch_norm(h)

#h = tf.nn.relu(tf.nn.conv2d_transpose(tf.reshape(h,[64,2,8,64]),ae5.weight['w3'],[64,2,8,64],[1,1,1,1],padding='SAME')+ae5.weight['b3'])
#h = batch_norm(h)
#h = tf.image.resize_nearest_neighbor(h,[4,16])
#############

#h = tf.nn.relu(tf.nn.conv2d_transpose(h,ae4.weight['w2'],[64,4,16,96],[1,1,1,1],padding='SAME')+ae4.weight['b2'])#ae4
#h = batch_norm(h)
#h = tf.image.resize_nearest_neighbor(h,[8,32])
#h = tf.nn.relu(tf.nn.conv2d_transpose(h,ae3.weight['w2'],[64,8,32,96],[1,1,1,1],padding='SAME')+ae3.weight['b2'])#ae3
#h = batch_norm(h)
#h = tf.image.resize_nearest_neighbor(h,[16,64])
#h = batch_norm(h)
h = tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(h,ae2.weight['w2'],[64,16,64,64],[1,1,1,1],padding='SAME')+ae2.weight['b2']))#ae2
h = tf.image.resize_bilinear(h,[32,128])
output = tf.nn.relu(tf.nn.conv2d(h,ae1.weight['w2'],[1,1,1,1],padding='SAME')+ae1.weight['b2'])#ae1

stackcost = tf.reduce_mean(tf.square(tf.subtract(x,output)))
opt = tf.train.AdamOptimizer(0.0185).minimize(stackcost)



data = Data('../data/data.npz', 64)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(0):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        _,cost = sess.run(ae1.partial_fit(),feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    #print(data.test([-1,32,128,3]))

    cost = sess.run(ae1.total_cost(),feed_dict={ae1.x:data.test([-1, 32, 128, 3])})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae1 train finished##################################")

for epoch in range(0):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        input = sess.run(ae1.encode,feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
        _,cost = sess.run(ae2.partial_fit(),feed_dict={ae2.x:input})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    input1 = sess.run(ae1.filture(), feed_dict={ae1.x: data.test([-1, 32, 128, 3])})
    cost = sess.run(ae2.total_cost(), feed_dict={ae2.x: input1})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae2 train finished##################################")


for epoch in range(0):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        input = sess.run(ae1.encode,feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
        input = sess.run(ae2.encode,feed_dict={ae2.x:input})

        _,cost = sess.run(ae3.partial_fit(),feed_dict={ae3.x:input})
        avg_cost += cost / len(data.train_data) * 64
    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    input = sess.run(ae1.filture(), feed_dict={ae1.x: data.test([-1, 32, 128, 3])})
    input = sess.run(ae2.filture(), feed_dict={ae2.x: input})
    cost = sess.run(ae3.total_cost(), feed_dict={ae3.x: input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae3 train finished##################################")

for epoch in range(0):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        input = sess.run(ae1.encode,feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
        input = sess.run(ae2.encode,feed_dict={ae2.x:input})
        input = sess.run(ae3.encode, feed_dict={ae3.x: input})
        _,cost = sess.run(ae4.partial_fit(),feed_dict={ae4.x:input})
        avg_cost += cost / len(data.train_data) * 64
    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    input = sess.run(ae1.filture(), feed_dict={ae1.x: data.test([-1, 32, 128, 3])})
    input = sess.run(ae2.filture(), feed_dict={ae2.x: input})
    input = sess.run(ae3.filture(), feed_dict={ae3.x: input})
    cost = sess.run(ae4.total_cost(), feed_dict={ae4.x: input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae4 train finished##################################")


for epoch in range(0):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        input = sess.run(ae1.filture(),feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
        input = sess.run(ae2.filture(),feed_dict={ae2.x:input})
        input = sess.run(ae3.filture(), feed_dict={ae3.x: input})
        input = sess.run(ae4.filture(), feed_dict={ae4.x: input})
        _,cost = sess.run(ae5.partial_fit(),feed_dict={ae5.x:input})
        avg_cost += cost / len(data.train_data) * 64
    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    input = sess.run(ae1.encode, feed_dict={ae1.x: data.test([-1, 32, 128, 3])})
    input = sess.run(ae2.encode, feed_dict={ae2.x: input})
    input = sess.run(ae3.encode, feed_dict={ae3.x: input})
    input = sess.run(ae4.encode, feed_dict={ae4.x: input})

    cost = sess.run(ae5.total_cost(), feed_dict={ae5.x: input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae5 train finished##################################")







for epoch in range(2000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        #input = sess.run(h,feed_dict={x:})
        _, cost = sess.run((opt,stackcost), feed_dict={x:data.batch_size([-1, 32, 128, 3])})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
