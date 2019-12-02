import argparse
import importlib
import os
import time

import matplotlib
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.keras.datasets import mnist

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from numpy import linalg, array, arange
from dpgan.utilize import normlization, Rsample
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, z_sampler, data, model,
                 sigma, digit, reg, lr, cilpc, batch_size,
                 num_batches, plot_size, save_size, d_iters,
                 data_name, data_path, path_output):  # changed
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.sigma = sigma
        self.digit = digit
        self.regc = reg
        self.lr = lr
        self.cilpc = cilpc
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.plot_size = plot_size
        self.save_size = save_size
        self.d_iters = d_iters
        self.data_name = data_name
        self.data_path = data_path
        self.path_output = path_output

        (self.data_td, self.label_td), (_, _) = mnist.load_data()
        self.data_td = np.reshape(self.data_td, (-1, 784))

        self.data_td = normlization(self.data_td)

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')  # [None, 784]
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(self.regc),
            weights_list=[var for var in tf.all_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)  # DP case
            grads_and_vars = self.d_rmsprop.compute_gradients(-1 * self.d_loss_reg, var_list=self.d_net.vars)
            dp_grads_and_vars = []  # noisy version
            for gv in grads_and_vars:  # for each pair
                g = gv[
                    0]  # get the gradient, type in loop one: Tensor("gradients/AddN_37:0", shape=(4, 4, 1, 64), dtype=float32)
                # print g # shape of all vars
                if g is not None:  # skip None case
                    g = self.dpnoise(g, self.batch_size)
                dp_grads_and_vars.append((g, gv[1]))
            self.d_rmsprop_new = self.d_rmsprop.apply_gradients(dp_grads_and_vars)  # should assign to a new optimizer
            # self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr) \
            #     .minimize(-1*self.d_loss_reg, var_list=self.d_net.vars) # non-DP case
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr) \
                .minimize(-1 * self.g_loss_reg, var_list=self.g_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -1 * self.cilpc, self.cilpc)) for v in self.d_net.vars]
        self.d_net_var_grad = [i for i in tf.gradients(self.d_loss_reg, self.d_net.vars) if
                               i is not None]  # explore the effect of noise on norm of D net variables's gradient vector, also remove None type
        self.norm_d_net_var_grad = []
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.g_loss_store = []  # store loss of generator
        self.d_loss_store = []  # store loss of discriminator
        self.wdis_store = []  # store Wasserstein distance, new added

    def train(self):  # batch_size*ite should be euough to use whole dataset for
        plt.ion()
        self.sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for t in range(0, self.num_batches):
            self.d_iters = 5
            if t % 500 == 0 or t < 25:  # make the discriminator more accurate at certain iterations
                self.d_iters = 100

            for _ in range(0, self.d_iters):  # train discriminator
                # data_td, label_td = self.x_sampler(self.batch_size) # data_td: data for training discriminator, data_td.shape: (self.batch_size, 784)
                data_td, label_td = Rsample(self.data_td, self.label_td, self.batch_size)
                bz = self.z_sampler(self.batch_size, self.z_dim)
                self.sess.run(self.d_rmsprop_new, feed_dict={self.x: data_td, self.z: bz})  # DP case
                # self.sess.run(self.d_rmsprop, feed_dict={self.x: data_td, self.z: bz}) # non-DP case
                self.sess.run(self.d_clip)

            bz = self.z_sampler(self.batch_size, self.z_dim)  # train generator, another batch of z sample
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: data_td})

            if t % self.plot_size == 0:  # evaluate loss and norm of gradient vector
                # bx,l = self.x_sampler(self.batch_size) # the reason we generate another batch of sample is that we want to see if the distance of 2 distributions are indeed pulled closer
                bx, l = Rsample(self.data_td, self.label_td, self.batch_size)

                bz = self.z_sampler(self.batch_size, self.z_dim)

                rd_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                rg_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx}
                )
                d_net_var_grad_val = self.sess.run(self.d_net_var_grad, feed_dict={self.x: bx, self.z: bz})
                if type(d_net_var_grad_val) != type([0]):
                    d_net_var_grad_val = [d_net_var_grad_val]
                self.norm_d_net_var_grad.append(self.norm_w(d_net_var_grad_val))
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t, time.time() - start_time, rd_loss, rg_loss))

                # store rd_loss, rg_loss and W-dis, new added
                self.g_loss_store.append(rg_loss)  # g_loss will decrease, here is not self.g_loss nor self.g_loss_reg
                self.d_loss_store.append(rd_loss)  # d_loss will increase
                self.wdis_store.append(rd_loss)  # Wasserstein distance will decrease

                # generate image
                bz = self.z_sampler(1, self.z_dim)  # changed, only generate 1 image
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})  # bx.shape: (1, 784)
                bx = np.reshape(bx, (1, 28, 28, 1))
                plt.figure(self.data + '.' + self.model)
                plt.imshow(bx.squeeze(), cmap="gray")
                plt.show()

            # if t % self.save_size == 0:  # store generator and discriminator, new added
            #     saver = tf.train.Saver()
            #     save_path = saver.save(self.sess, "result/sesssave/sess.ckpt")
            #     print("Session saved in file: %s" % save_path)

        z_sample = self.z_sampler(len(self.label_td), self.z_dim)
        x_gene = self.sess.run(self.x_, feed_dict={
            self.z: z_sample})  # type(x_gene): <type 'numpy.ndarray'>, x_gene[0].shape: (784,)
        x_gene = array(x_gene) * 255  # to 0-255 scale

        # store generated data
        with open(self.path_output + 'datafile/x_gene_' + self.digit + '_sigma' + str(self.sigma) + '.pickle', 'wb') as fp:
            pickle.dump(x_gene, fp)

        # store wdis
        t = arange(len(self.wdis_store))
        plt.plot(t, self.wdis_store, 'b--')
        plt.xlabel('Generator iterations')
        plt.ylabel('Wasserstein distance')
        plt.savefig('result/lossfig/wdis.jpg')
        plt.close()

        with open(self.path_output + 'lossfile/wdis.pickle', 'wb') as fp:
            pickle.dump(self.wdis_store, fp)

        # store generator and discriminator
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.path_output + 'sesssave/sess.ckpt')
        print("Training finished, session saved in file: %s" % save_path)

    def dpnoise(self, tensor, batch_size):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        rt = tf.random_normal(s, mean=0.0, stddev=self.sigma)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batch_size), rt))
        return t

    def norm_w(self, v):
        return sum([linalg.norm(i) for i in v])


def train_on_mnist():
    data = mnist
    model = importlib.import_module("mnist.mlp")

    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()

    digits = ['2', '3', '4', '5']  # MNIST digits need to use
    for digit in digits:
        tf.reset_default_graph()

        d_net = model.Discriminator()  # mnist/mlp.py, d_net is a instance of class Discriminator
        g_net = model.Generator()

        sigma_all = 0  # total noise std added (800 before)
        reg = 2.5e-5
        lr = 5e-5
        cilpc = 0.5  # 0.02
        batch_size = 64
        num_batches = 10000  # 150000
        plot_size = 5
        save_size = 100000
        d_iters = 5
        data_name = 'training'
        data_path = "/mnist/MNIST"
        path_output = "/"

        wgan = WassersteinGAN(g_net, d_net, zs, "mnist", "mlp", sigma_all,
                              digit, reg, lr, cilpc, batch_size, num_batches,
                              plot_size, save_size, d_iters, data_name, data_path,
                              path_output)
        wgan.train()

if __name__ == '__main__':
    train_on_mnist()
