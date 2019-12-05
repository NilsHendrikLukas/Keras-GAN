from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

from emnist import extract_training_samples

import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
from sklearn.utils import shuffle


class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

        xin, xout, test = self.load_mnist_data(n_xin=40000, n_xout=10000)

    def load_mnist_data(self,
                        n_xin,
                        n_xout):
        """
        Load x_in, x_out and the test set
        @:param n_xin: Size of the X_in dataset
        @:param n_xout: Size of the X_out dataset
        @:return xin, xout, test
        """
        def normalize(data):
            return np.reshape((data.astype(np.float32) - 127.5) / 127.5, (-1, 28, 28, 1))

        # Load and normalize the training data
        (x_train, y_train), (x_test, y_test) =  extract_training_samples('digits')
        x_train, x_test = normalize(x_train), normalize(x_test)

        # Shuffle for some randomness
        x_train, y_train = shuffle(x_train, y_train)

        assert(n_xin+n_xout < len(x_train))  # No overflow, sizes have to be assured

        # Split into x_in and x_out
        x_in, y_in = x_train[:n_xin], y_train[:n_xin]
        x_out, y_out = x_test[n_xin:n_xin+n_xout], y_test[n_xin:n_xin+n_xout]

        return (x_in, y_in), (x_out, y_out), (x_test, x_test)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        """ Builds a generator for MNIST
        """
        noise = Input(128 * 7 * 7)

        l0 = Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim)(noise)
        l1 = Reshape((7, 7, 128))(l0)
        l2 = UpSampling2D()(l1)
        l3 = Conv2D(128, kernel_size=4, padding="same")(l2)
        l4 = BatchNormalization(momentum=0.8)(l3)
        l5 = Activation("relu")(l4)
        l6 = UpSampling2D()(l5)
        l7 = Conv2D(64, kernel_size=4, padding="same")(l6)
        l8 = BatchNormalization(momentum=0.8)(l7)
        l9 = Activation("relu")(l8)
        l10 = Conv2D(self.channels, kernel_size=4, padding="same")(l9)
        generator_out = Activation("tanh")(l10)

        return Model(noise, generator_out)

    def build_critic(self):
        """ Builds the critic for an MNIST model
        """
        critic_in = Input((28, 28, 1))

        l0 = Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(critic_in)
        l1 = LeakyReLU(alpha=0.2)(l0)
        l2 = Dropout(0.25)(l1)
        l3 = Conv2D(32, kernel_size=3, strides=2, padding="same")(l2)
        l4 = ZeroPadding2D(padding=((0, 1), (0, 1)))(l3)
        l5 = BatchNormalization(momentum=0.8)(l4)
        l6 = LeakyReLU(alpha=0.2)(l5)
        l7 = Dropout(0.25)(l6)
        l8 = Conv2D(64, kernel_size=3, strides=2, padding="same")(l7)
        l9 = BatchNormalization(momentum=0.8)(l8)
        l10 = LeakyReLU(alpha=0.2)(l9)
        l11 = Dropout(0.25)(l10)
        l12 = Conv2D(128, kernel_size=3, strides=1, padding="same")(l11)
        l13 = BatchNormalization(momentum=0.8)(l12)
        l14 = LeakyReLU(alpha=0.2)(l13)
        l15 = Dropout(0.25)(l14)
        featuremaps = Flatten()(l15)
        critic_out = Dense(1, name="critic_out")(featuremaps)

        return Model(critic_in, critic_out)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)