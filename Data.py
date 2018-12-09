import numpy as np

class Data(object):
    def __init__(self, datafile, batchsize):
        self.img = np.load(datafile)['train']
        self.train = self.img[:4480]
        self.train_index = np.arange(len(self.train))

        self.batchsize = batchsize

    def train_batch_size(self, resize):
        np.random.shuffle(self.train_index)
        self.train_data = self.train[self.train_index, :]
        return np.reshape(self.train_data[:self.batchsize],resize)

