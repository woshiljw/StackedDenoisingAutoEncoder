import numpy as np
import tensorflow as tf
from autoencoders.AutoEncoder import Autoencoder_conv2conv, Autoencoder_conv2deconv, Autoencoder_full2deconv
from Data import Data
from tensorflow.contrib.layers import batch_norm
import cv2
from tensorflow.python.training import moving_averages

data = Data('../data/data.npz', 64)#读入数据

#定义第一层自动ＤＡＥ
ae1 = Autoencoder_conv2conv('ae1',
                              [5,5,3,64],#输入卷积核的大小
                              [1,1,64,3],#输出卷积核的大小
                              [64,32,128,3],#输入该层的数据的形状
                              tf.train.AdamOptimizer(0.85))

#定义第二层自动ＤＡＥ
ae2 = Autoencoder_conv2deconv('ae2',
                              [3,3,64,96],#输入卷积核的大小
                              [5,5,64,96],#输出卷积核的大小
                              [64,32,128,64],#输入该层的数据的形状
                              tf.train.AdamOptimizer(0.85))

#定义第三层自动ＤＡＥ
ae3 = Autoencoder_conv2deconv('ae3',
                              [3,3,96,96],#输入卷积核的大小
                              [3,3,96,96],#输出卷积核的大小
                              [64,16,64,96],#输入该层的数据的形状
                              tf.train.AdamOptimizer(0.85))

#定义第四层自动ＤＡＥ
ae4 = Autoencoder_conv2deconv('ae4',
                              [3,3,96,64],#输入卷积核的大小
                              [3,3,96,64],#输出卷积核的大小
                              [64,8,32,96],#输入该层的数据的形状
                              tf.train.AdamOptimizer(0.85))

#定义全连接层提取出64维参数
ae5 = Autoencoder_full2deconv('ae5',
                              tf.train.AdamOptimizer(0.185))

'''
由于将图像特征提取到64维后训练结果很差，因此未使用ae5

'''

#定义ｂａｔｃｈｎｏｒｍ层
def batch_normalization_layer(input,moving_mean,moving_variance,beta,gamma):
    axis = list(range(len(input.get_shape())-1))
    mean,variance = tf.nn.moments(input,axis)#计算当前epoch的均值与方差
    moving_averages.assign_moving_average(moving_mean,mean,0.999)#修改滑动均值
    moving_averages.assign_moving_average(moving_variance,variance,0.999)#修改滑动方差
    '''计算滑动均值和滑动方差是为了在测试集上使用，使用未知样本进行计算时，使用滑动均值和滑动方差来代替未知样本的均值和方差'''
    return tf.nn.batch_normalization(input,mean,variance,beta,gamma,0.001)


'''
定义ＳＤＡＥ的计算图（该部分只是定义图，并未训练！！！）
该部分使用之前定义的ＤＡＥ的参数来继续训练
'''
#autoencoder1
'''
卷积核的参数保存在类的weights字典里,batchnorm层的参数保存在bnparam内，具体方式定义在类里
'''
y = tf.placeholder(tf.float32,[64,32,128,3])
x = tf.placeholder(tf.float32,[64,32,128,3])
h = tf.nn.conv2d(x,ae1.weight['w1'],[1,1,1,1],padding='SAME')+ae1.weight['b1']
h = batch_normalization_layer(h,ae1.bnparam1['moving_mean'],ae1.bnparam1['moving_variance'],
                              ae1.bnparam1['beta'],ae1.bnparam1['gamma'])
h = tf.nn.relu(h)
#autoencoder2
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = tf.nn.conv2d(h,ae2.weight['w1'],[1,1,1,1],padding='SAME')+ae2.weight['b1']
h = batch_normalization_layer(h,ae2.bnparam1['moving_mean'],ae2.bnparam1['moving_variance'],
                              ae2.bnparam1['beta'],ae2.bnparam1['gamma'])
h = tf.nn.relu(h)
#autoencoder3
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = tf.nn.conv2d(h,ae3.weight['w1'],[1,1,1,1],padding='SAME')+ae3.weight['b1']
h = batch_normalization_layer(h,ae3.bnparam1['moving_mean'],ae3.bnparam1['moving_variance'],
                              ae3.bnparam1['beta'],ae3.bnparam1['gamma'])
h = tf.nn.relu(h)

#autoencoder4
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = tf.nn.conv2d(h,ae4.weight['w1'],[1,1,1,1],padding='SAME')+ae4.weight['b1']#ae4
h = batch_normalization_layer(h,ae4.bnparam1['moving_mean'],ae4.bnparam1['moving_variance'],
                              ae4.bnparam1['beta'],ae4.bnparam1['gamma'])
h = tf.nn.relu(h)
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

h = tf.nn.conv2d_transpose(h,ae4.weight['w2'],[64,8,32,96],[1,2,2,1],padding='SAME')+ae4.weight['b2']
h = batch_normalization_layer(h,ae4.bnparam2['moving_mean'],ae4.bnparam2['moving_variance'],
                              ae4.bnparam2['beta'],ae4.bnparam2['gamma'])
h = tf.nn.relu(h)
#autoencoder4
h = tf.nn.conv2d_transpose(h,ae3.weight['w2'],[64,16,64,96],[1,2,2,1],padding='SAME')+ae3.weight['b2']
h = batch_normalization_layer(h,ae3.bnparam2['moving_mean'],ae3.bnparam2['moving_variance'],
                              ae3.bnparam2['beta'],ae3.bnparam2['gamma'])
h = tf.nn.relu(h)
#autoencoder3

h = tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(h,ae2.weight['w2'],[64,32,128,64],[1,2,2,1],padding='SAME')+ae2.weight['b2']))#ae2
h = batch_normalization_layer(h,ae2.bnparam2['moving_mean'],ae2.bnparam2['moving_variance'],
                              ae2.bnparam2['beta'],ae2.bnparam2['gamma'])
h = tf.nn.relu(h)
#autoencoder2
h = tf.nn.relu(tf.nn.conv2d(h,ae1.weight['w2'],[1,1,1,1],padding='SAME')+ae1.weight['b2'])#ae1
h = batch_normalization_layer(h,ae1.bnparam2['moving_mean'],ae1.bnparam2['moving_variance'],
                              ae1.bnparam2['beta'],ae1.bnparam2['gamma'])
output = tf.nn.relu(h)
#autoencoder1
stackcost = tf.reduce_mean(tf.square(tf.subtract(y,output)))
opt = tf.train.AdamOptimizer(0.185).minimize(stackcost)




sess = tf.Session()
sess.run(tf.global_variables_initializer())

#训练AutoEncoder1
for epoch in range(200):
    avg_cost = 0#定义平均ｃｏｓｔ
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    gaussianNoise = 0.01 * np.random.normal(size=[64, 12288]).reshape([64, 32, 128, 3])#0.01高斯噪声
    for i in range(total_batch):
        traindata = data.batch_size([64, 32, 128, 3])#读取[64, 32, 128, 3]大小的数据

        _,cost = sess.run(ae1.partial_fit(),feed_dict={ae1.x:traindata+gaussianNoise,ae1.y:traindata})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
#测试AutoEncoder1
for i in range(total_batch):
    #print(data.test([-1,32,128,3]))
    testdata = data.test([-1, 32, 128, 3])
    cost = sess.run(ae1.total_cost(),feed_dict={ae1.x:testdata,ae1.y:testdata})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae1 train finished##################################")

for epoch in range(200):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    gaussianNoise = 0.01 * np.random.normal(size=[64,32,128,64])
    for i in range(total_batch):
        traindata = data.batch_size([-1, 32, 128, 3])
        input = sess.run(ae1.encode,feed_dict={ae1.x:traindata,ae1.y:traindata})

        _,cost = sess.run(ae2.partial_fit(),feed_dict={ae2.x:input+gaussianNoise,ae2.y:input})
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    testdata = data.test([-1, 32, 128, 3])
    input = sess.run(ae1.filture(), feed_dict={ae1.x: testdata,ae1.y: testdata})
    #input = batch_norm(input)
    cost = sess.run(ae2.total_cost(), feed_dict={ae2.x: input,ae2.y: input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae2 train finished##################################")


for epoch in range(200):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    gaussianNoise = 0.01 * np.random.normal(size=[64,16,64,96])
    for i in range(total_batch):
        traindata = data.batch_size([-1, 32, 128, 3])
        input = sess.run(ae1.encode, feed_dict={ae1.x: traindata, ae1.y: traindata})
        input = sess.run(ae2.encode,feed_dict={ae2.x:input,ae2.y:input})

        _,cost = sess.run(ae3.partial_fit(),feed_dict={ae3.x:input+gaussianNoise,ae3.y:input})
        avg_cost += cost / len(data.train_data) * 64
    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    testdata = data.test([-1, 32, 128, 3])
    input = sess.run(ae1.filture(), feed_dict={ae1.x: testdata,ae1.y:testdata})
    #input = batch_norm(input)
    input = sess.run(ae2.filture(), feed_dict={ae2.x: input,ae2.y: input})
    cost = sess.run(ae3.total_cost(), feed_dict={ae3.x: input,ae3.y:input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae3 train finished##################################")

for epoch in range(200):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    gaussianNoise = 0.01 * np.random.normal(size=[64,8,32,96])
    for i in range(total_batch):
        traindata = data.batch_size([-1, 32, 128, 3])
        input = sess.run(ae1.encode, feed_dict={ae1.x: traindata, ae1.y: traindata})
        input = sess.run(ae2.encode, feed_dict={ae2.x: input, ae2.y: input})
        input = sess.run(ae3.encode, feed_dict={ae3.x: input,ae3.y: input})

        _,cost = sess.run(ae4.partial_fit(),feed_dict={ae4.x:input+gaussianNoise,ae4.y:input})
        avg_cost += cost / len(data.train_data) * 64
    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
avg_cost = 0
total_batch = int(len(data.test_data) / 64)
data.num = 0
for i in range(total_batch):
    testdata = data.test([-1, 32, 128, 3])
    input = sess.run(ae1.filture(), feed_dict={ae1.x: testdata,ae1.y: testdata})
    input = sess.run(ae2.filture(), feed_dict={ae2.x: input,ae2.y: input})
    input = sess.run(ae3.filture(), feed_dict={ae3.x: input,ae3.y: input})
    cost = sess.run(ae4.total_cost(), feed_dict={ae4.x: input,ae4.y: input})
    avg_cost += cost / len(data.test_data) * 64
print("test cost: ",avg_cost)
print("#################################ae4 train finished##################################")


# for epoch in range(0):
#     avg_cost = 0
#     total_batch = int(len(data.train_data) / 64)
#     data.num = 0
#     for i in range(total_batch):
#         input = sess.run(ae1.filture(),feed_dict={ae1.x:data.batch_size([-1, 32, 128, 3])})
#         #input = batch_norm(input)
#         input = sess.run(ae2.filture(),feed_dict={ae2.x:input})
#         input = sess.run(ae3.filture(), feed_dict={ae3.x: input})
#         input = sess.run(ae4.filture(), feed_dict={ae4.x: input})
#         _,cost = sess.run(ae5.partial_fit(),feed_dict={ae5.x:input})
#         avg_cost += cost / len(data.train_data) * 64
#     print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
# avg_cost = 0
# total_batch = int(len(data.test_data) / 64)
# data.num = 0
# for i in range(total_batch):
#     input = sess.run(ae1.encode, feed_dict={ae1.x: data.test([-1, 32, 128, 3])})
#     #input = batch_norm(input)
#     input = sess.run(ae2.encode, feed_dict={ae2.x: input})
#     input = sess.run(ae3.encode, feed_dict={ae3.x: input})
#     input = sess.run(ae4.encode, feed_dict={ae4.x: input})
#
#     cost = sess.run(ae5.total_cost(), feed_dict={ae5.x: input})
#     avg_cost += cost / len(data.test_data) * 64
# print("test cost: ",avg_cost)
# print("#################################ae5 train finished##################################")
# #





'''最终训练SDAE'''
for epoch in range(20000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)
    data.num = 0
    for i in range(total_batch):
        #input = sess.run(h,feed_dict={x:})
        traindata = data.batch_size([-1, 32, 128, 3])
        gaussianNoise = 0.01 * np.random.normal(size=np.shape(traindata))
        _, cost = sess.run((opt,stackcost), feed_dict={x:traindata+gaussianNoise,y:traindata})
        if epoch%100==0 and i%64==0:
            '''
            每100个epoch保存一个输出的图像
            '''
            rebuildimage = sess.run(output,feed_dict={x:traindata})
            cv2.imwrite('./outputImage/input.hdr', traindata[0][:, :, ::-1])
            #print(output.shape)
            cv2.imwrite('./outputImage/' + str(epoch) + '_output.hdr',
                        np.reshape(rebuildimage[0], [32, 128, 3])[:, :, ::-1])
        avg_cost += cost / len(data.train_data) * 64

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
