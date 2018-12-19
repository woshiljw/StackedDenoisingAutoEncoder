import numpy as np

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
        self.num += self.batchsize
        return np.reshape(self.train_data[self.num - self.batchsize:self.num], resize)

    def test(self,resize):
        self.num += self.batchsize
        return np.reshape(self.test_data[self.num - self.batchsize:self.num], resize)

