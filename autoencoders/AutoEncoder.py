import tensorflow as tf

#两个卷积层组成的自编码器
class Autoencoder_conv2conv(object):
    def __init__(self,name,encoder_filter_size,decoder_filter_size
                 ,input_shape,optimizer,
                 transfer_function=tf.nn.relu):
        self.weight = self._initialize_weights(name,encoder_filter_size,decoder_filter_size)
        self.x = tf.placeholder(tf.float32,input_shape)

        #编码解码部分
        self.encode = transfer_function(
            tf.add(
                tf.nn.conv2d(self.x,self.weight['w1'],[1,1,1,1],padding="SAME"),
                self.weight['b1']
            )
        )
        self.decode = transfer_function(
            tf.add(
                tf.nn.conv2d(self.encode,self.weight['w2'],[1,1,1,1],padding="SAME"),
                self.weight['b2']
            )
        )

        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.decode,self.x)))
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
                                            shape=[decoder_filter_size[3]],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return all_weights

    def partial_fit(self):
        return (self.optimizer,self.cost)

    def total_cost(self):
        return self.cost


#卷积层与反卷积层组成的自编码器
class Autoencoder_conv2deconv(object):
    def __init__(self,name,encoder_filter_size,decoder_filter_size,
                 input_shape,optimizer,transfer_function=tf.nn.relu):
        self.weight = self._initialize_weights(name,encoder_filter_size,decoder_filter_size)
        self.x = tf.placeholder(tf.float32,input_shape)

        #编码解码部分
        self.maxpool = tf.nn.max_pool(self.x,[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        self.encode = transfer_function(
            tf.add(
                tf.nn.conv2d(self.maxpool,self.weight['w1'],strides=[1,1,1,1],padding='SAME'),
                self.weight['b1']
            )
        )

        decoder_output_shape = [input_shape[0],input_shape[1]//2,input_shape[2]//2,input_shape[3]]
        self.decode = transfer_function(
            tf.add(
                tf.nn.conv2d_transpose(self.encode,self.weight['w2'],decoder_output_shape,[1,1,1,1],padding='SAME'),
                self.weight['b2']
            )
        )

        self.upscale = tf.image.resize_nearest_neighbor(self.decode,input_shape[1:3])
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.upscale, self.x)))
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

    def partial_fit(self):
        return (self.optimizer,self.cost)

    def total_cost(self):
        return self.cost

#全连接层与反卷积层组成的自编码器
class Autoencoder_full2deconv(object):
    def __init__(self):
        pass