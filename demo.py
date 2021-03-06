import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers import batch_norm

class Data(object):
    '''
    Data类，用于读入ｎｐｚ格式数据，并根据给定epoch大小和形状返回训练数据
    '''
    def __init__(self, datafile, batchsize):
        '''
        :param datafile: npz数据集文件路径
        :param batchsize: 一个batch数据的数量，该部分写死了只能用64，应为在反卷积层中需要输入batch大小从卷积层读出batch大小为None，输入到反卷积层会报错，所以这部分就写死了
        '''
        self.img = np.load(datafile)['train']
        self.train = self.img[:4480]#前4480个数据用于训练样本
        self.train_index = np.arange(len(self.train))#获取训练样本索引
        np.random.shuffle(self.train_index)#打乱索引
        self.train_data = self.train[self.train_index, :]#打乱训练数据
        self.batchsize = batchsize
        self.test_ = self.img[4480:]
        self.test_index = np.arange(len(self.test_))
        np.random.shuffle(self.test_index)
        self.test_data = self.test_[self.test_index,:]
        self.num = 0


    def batch_size(self, resize):
        '''
        :param resize: tensor的大小，默认为［64，32，128，3］
        :return: 返回训练数据
        '''
        self.num+=self.batchsize
        return np.reshape(self.train_data[self.num-self.batchsize:self.num], resize)

    def test(self,resize):
        return np.reshape(self.test_data[0:self.batchsize], resize)

    def shuffle(self):
        np.random.shuffle(self.train_index)  # 打乱索引
        self.train_data = self.train[self.train_index, :]
        self.num = 0

def batch_normalization_layer(input,name):
    axis = list([0,1,2])
    mean,variance = tf.nn.moments(input,axis)
    beta = tf.get_variable('beta'+name,initializer=tf.zeros_initializer,shape=input.get_shape()[-1],dtype=tf.float32)
    gamma = tf.get_variable('gamma'+name,initializer=tf.ones_initializer,shape=input.get_shape()[-1],dtype=tf.float32)

    # moving_mean = tf.get_variable('moving_mean'+name,initializer=tf.zeros_initializer,
    #                               shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    # moving_variance = tf.get_variable('moving_var'+name,initializer=tf.zeros_initializer,
    #                                   shape=input.get_shape()[-1],dtype=tf.float32,trainable=False)
    # update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,0.999)
    # update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,0.999)
    #
    # tf.add_to_collection('mean'+name,update_moving_mean)
    # tf.add_to_collection('variance'+name,update_moving_variance)

    return tf.nn.batch_normalization(input,mean,variance,beta,gamma,0.001)



y = tf.placeholder(tf.float32, [64, 32, 128, 3])

x = tf.placeholder(tf.float32, [64, 32, 128, 3],name='input')
h = tf.layers.conv2d(x,64,[5,5],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.layers.max_pooling2d(h,[2,2],[2,2])


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')


h = tf.layers.conv2d(h,96,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')


h = tf.layers.conv2d(h,64,[3,3],[1,1],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')



h = tf.layers.flatten(h)
encode = tf.layers.dense(h,64)
h = batch_norm(encode)
h = tf.nn.relu(h)

h = tf.layers.dense(h,1024)
h = batch_norm(h)
h = tf.nn.relu(h)
h = tf.reshape(h,[64,2,8,64])

h = tf.layers.conv2d_transpose(h,64,[3,3],[2,2],padding='SAME')
# decode = tf.layers.conv2d(encode,64,[3,3],[2,2],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
# h = tf.layers.conv2d_transpose(h,64,[3,3],[2,2],padding='SAME')


h = tf.layers.conv2d_transpose(h,96,[3,3],[2,2],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
# h = tf.layers.conv2d_transpose(h,96,[3,3],[2,2],padding='SAME')


h = tf.layers.conv2d_transpose(h,96,[3,3],[2,2],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
# h = tf.layers.conv2d_transpose(h,96,[3,3],[2,2],padding='SAME')


h = tf.layers.conv2d_transpose(h,64,[5,5],[2,2],padding='SAME')
h = batch_norm(h)
h = tf.nn.relu(h)
# h = tf.layers.conv2d_transpose(h,64,[5,5],[2,2],padding='SAME')


out = tf.layers.conv2d(h,3,[1,1],[1,1],padding='SAME')
# out = batch_norm(out)
# out = tf.nn.relu(out)

saver = tf.train.Saver()

tf.add_to_collection('encode',encode)
# tf.add_to_collection('decode',decode)
tf.add_to_collection('reconstruct',out)

stackcost = tf.reduce_mean(tf.square(tf.subtract(y, out)))
opt = tf.train.AdamOptimizer(0.00185).minimize(stackcost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

data = Data('../data/data.npz', 64)
for epoch in range(20000):
    avg_cost = 0
    total_batch = int(len(data.train_data) / 64)

    data.num=0
    for i in range(total_batch):
        # input = sess.run(h,feed_dict={x:})
        gaussianNoise = 0.01*np.random.normal(size=[64,12288]).reshape([64,32,128,3])

        traindata = data.batch_size([-1, 32, 128, 3])
        _, cost = sess.run((opt, stackcost), feed_dict={x: traindata+gaussianNoise,y:traindata})

        avg_cost += cost / len(data.train_data) * 64
        if i % 50 == 0 and epoch %50==0:
            rebuildimage = sess.run(out, feed_dict={x: data.test([64,32,128,3])})
            cv2.imwrite('./outputImage/input.hdr', data.test([64,32,128,3])[0][:, :, ::-1])
            # print(output.shape)
            # print(rebuildimage.shape)
            # print(rebuildimage.shape)
            cv2.imwrite('./outputImage/' + str(epoch) + '_output.hdr',
                        np.reshape(rebuildimage[0], [32, 128, 3])[:, :, ::-1])

    if epoch % 100 == 0:
        saver.save(sess,'./saveModel/my_model',global_step=epoch)

    # if epoch % 1000 == 0:
    #     data.shuffle()

    print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))
