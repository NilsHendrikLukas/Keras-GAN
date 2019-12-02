import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.keras.datasets import mnist

class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        self.train = x_train, y_train


    def __call__(self, batch_size): # __call__ method is executed when the instance is called
        return self.train[0][:batch_size], self.train[1][:batch_size]
        # mnist_models.train.next_batch(batch_size) is a tuple with shape: (batch_size*784, batch_size), 2nd component is the row array for label
        # the data here is already normlized (/255)
        # type(mnist_models.train.next_batch(2)[0][0][0]): numpy.float32

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]) # "changed"
        # the shape of return is: batch_size*z_dim