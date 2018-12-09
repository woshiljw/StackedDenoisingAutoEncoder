import numpy as np

class Data(object):
    def __init__(self, datafile, batchsize):
        self.img = np.load(datafile)['train']
        self.train = self.img[:4480]
        self.train_index = np.arange(len(self.train))
        np.random.shuffle(self.train_index)
        self.train_data = self.train[self.train_index, :]
        self.batchsize = batchsize
        self.num = 0

    def batch_size(self, resize):
        self.num += self.batchsize
        return np.reshape(self.train_data[self.num - self.batchsize:self.num], resize)