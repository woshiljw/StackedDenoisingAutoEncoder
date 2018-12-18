import tensorflow as tf
from tensorflow.python.training import moving_averages
#两个卷积层组成的自编码器
class Autoencoder_conv2conv(object):
    def __init__(self,name,encoder_filter_size,decoder_filter_size
                 ,input_shape,optimizer,
                 transfer_function=tf.nn.relu):
        self.weight = self._initialize_weights(name,encoder_filter_size,decoder_filter_size)
        self.x = tf.placeholder(tf.float32,input_shape)
        self.y = tf.placeholder(tf.float32, input_shape)
        self.bnparam1 = self._initial_batchparam('_in_'+name,[64,32,128,64])
        self.bnparam2 = self._initial_batchparam('_out_' + name, [64, 32, 128, 3])
        #编码解码部分
        self.encode = transfer_function(
            self.batch_normalization_layer(
            tf.add(
                tf.nn.conv2d(self.x,self.weight['w1'],[1,1,1,1],padding="SAME"),
                self.weight['b1']
            ),param=self.bnparam1)
        )

        self.decode = transfer_function(
            self.batch_normalization_layer(
            tf.add(
                tf.nn.conv2d(self.encode,self.weight['w2'],[1,1,1,1],padding="SAME"),
                self.weight['b2']
            )
            ,param=self.bnparam2)
        )

        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.decode,self.y)))
        self.optimizer = optimizer.minimize(self.cost)

    #权值初始化
    def _initialize_weights(self,name,encoder_filter_size,decoder_filter_size):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(name=name + '_w1',
                                            shape=encoder_filter_size,
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b1'] = tf.get_variable(name=name + '_b1',
                                            shape=[encoder_filter_size[3]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['w2'] = tf.get_variable(name=name + '_w2',
                                            shape=decoder_filter_size,
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b2'] = tf.get_variable(name=name + '_b2',
                                            shape=[decoder_filter_size[3]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return all_weights

    #batchnorm参数的初始化
    def _initial_batchparam(self,name,input_shape):
        param = dict()
        param['beta'] = tf.get_variable('beta' + name, initializer=tf.zeros_initializer, shape=input_shape[-1],
                               dtype=tf.float32)
        param['gamma'] = tf.get_variable('gamma' + name, initializer=tf.ones_initializer, shape=input_shape[-1],
                                dtype=tf.float32)
        param['moving_mean'] = tf.get_variable('moving_mean' + name, initializer=tf.zeros_initializer,
                                           shape=input_shape[-1], dtype=tf.float32, trainable=False)
        param['moving_variance'] = tf.get_variable('moving_var' + name, initializer=tf.zeros_initializer,
                                               shape=input_shape[-1], dtype=tf.float32, trainable=False)
        return param

    #batchnorm层
    def batch_normalization_layer(self,input,param):
        axis = list(range(len(input.get_shape()) - 1))
        mean, variance = tf.nn.moments(input, axis)
        moving_averages.assign_moving_average(param['moving_mean'], mean, 0.999)
        moving_averages.assign_moving_average(param['moving_variance'], variance, 0.999)

        return tf.nn.batch_normalization(input, mean, variance, param['beta'], param['gamma'], 0.001)

    def partial_fit(self):
        return (self.optimizer,self.cost)

    def total_cost(self):
        return self.cost

    def filture(self):
        return self.encode

#卷积层与反卷积层组成的自编码器
class Autoencoder_conv2deconv(object):
    def __init__(self,name,encoder_filter_size,decoder_filter_size,
                 input_shape,optimizer,transfer_function=tf.nn.relu):
        self.weight = self._initialize_weights(name,encoder_filter_size,decoder_filter_size)
        self.x = tf.placeholder(tf.float32,input_shape)
        self.y = tf.placeholder(tf.float32,input_shape)
        self.bnparam1 = self._initial_batchparam('_in_' + name, encoder_filter_size)



        #编码解码部分
        self.input = self.x
        self.maxpool = tf.nn.max_pool(self.input,[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        self.encode = transfer_function(
            self.batch_normalization_layer(
            tf.add(
                tf.nn.conv2d(self.maxpool,self.weight['w1'],strides=[1,1,1,1],padding='SAME'),
                self.weight['b1']
            )
            ,param=self.bnparam1)
        )

        decoder_output_shape = [input_shape[0],input_shape[1],input_shape[2],input_shape[3]]
        self.bnparam2 = self._initial_batchparam('_out_' + name, decoder_output_shape)
        self.decode = transfer_function(
            self.batch_normalization_layer(
            tf.add(
                tf.nn.conv2d_transpose(self.encode,self.weight['w2'],decoder_output_shape,[1,2,2,1],padding='SAME'),
                self.weight['b2']
            )
            ,param=self.bnparam2)
        )

        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.decode, self.y)))
        self.optimizer = optimizer.minimize(self.cost)

    def _initialize_weights(self,name,encoder_filter_size,decoder_filter_size):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(name=name + '_w1',
                                            shape=encoder_filter_size,
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b1'] = tf.get_variable(name=name + '_b1',
                                            shape=[encoder_filter_size[3]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['w2'] = tf.get_variable(name=name + '_w2',
                                            shape=decoder_filter_size,
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b2'] = tf.get_variable(name=name + '_b2',
                                            shape=[decoder_filter_size[2]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return all_weights

    def _initial_batchparam(self,name,input_shape):
        param = dict()
        param['beta'] = tf.get_variable('beta' + name, initializer=tf.zeros_initializer, shape=input_shape[-1],
                               dtype=tf.float32)
        param['gamma'] = tf.get_variable('gamma' + name, initializer=tf.ones_initializer, shape=input_shape[-1],
                                dtype=tf.float32)
        param['moving_mean'] = tf.get_variable('moving_mean' + name, initializer=tf.zeros_initializer,
                                           shape=input_shape[-1], dtype=tf.float32, trainable=False)
        param['moving_variance'] = tf.get_variable('moving_var' + name, initializer=tf.zeros_initializer,
                                               shape=input_shape[-1], dtype=tf.float32, trainable=False)
        return param

    def batch_normalization_layer(self,input,param):
        axis = list(range(len(input.get_shape()) - 1))
        mean, variance = tf.nn.moments(input, axis)
        moving_averages.assign_moving_average(param['moving_mean'], mean, 0.999)
        moving_averages.assign_moving_average(param['moving_variance'], variance, 0.999)

        return tf.nn.batch_normalization(input, mean, variance, param['beta'], param['gamma'], 0.001)

    def partial_fit(self):
        return (self.optimizer,self.cost)

    def total_cost(self):
        return self.cost
    def filture(self):
        return self.encode

#全连接层与反卷积层组成的自编码器
class Autoencoder_full2deconv(object):
    def __init__(self,name,optimizer,transfer_function=tf.nn.relu):
        self.weight = self._initialize_weights(name)
        self.x = tf.placeholder(tf.float32,[64,4,16,64])

        self.inp = tf.nn.max_pool(self.x,[1,2,2,1],[1,2,2,1],padding='SAME')
        self.input = tf.reshape(self.inp,[64,1024])

        self.fullconnect1 = transfer_function(
            tf.add(
                tf.matmul(self.input,self.weight['w1']),
                self.weight['b1']
            )
        )
        self.fullconnect2 = transfer_function(
            tf.add(
                tf.matmul(self.fullconnect1,self.weight['w2']),
                self.weight['b2']
            )
        )

        self.deconv = transfer_function(
            tf.add(
                tf.nn.conv2d_transpose(tf.reshape(self.fullconnect2,[64,2,8,64]),self.weight['w3'],[64,2,8,64],[1,1,1,1],padding='SAME'),
                self.weight['b3'])
        )
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.inp,self.deconv)))

        self.optimizer = optimizer.minimize(self.cost)


    def _initialize_weights(self,name):
        all_weights = dict()

        all_weights['w1'] = tf.get_variable(name=name+'_w1',
                                            shape=[2*8*64,64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b1'] = tf.get_variable(name=name+'_b1',
                                            shape=[64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['w2'] = tf.get_variable(name=name+'_w2',
                                            shape=[64,2*8*64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b2'] = tf.get_variable(name=name+'_b2',
                                            shape=[2*8*64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['w3'] = tf.get_variable(name=name+'_w3',
                                            shape=[3,3,64,64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        all_weights['b3'] = tf.get_variable(name=name+'_b3',
                                            shape=[64],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        return all_weights

    def partial_fit(self):
        return (self.optimizer,self.cost)
    def total_cost(self):
        return self.cost
