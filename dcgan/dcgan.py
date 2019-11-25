from __future__ import print_function, division

from keras import initializers
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        # Generator network
        model = Sequential()

        init = initializers.RandomNormal(stddev=0.02)
        # FC: 2x2x512
        model.add(Dense(2 * 2 * 512, input_shape=(self.latent_dim,), kernel_initializer=init))
        model.add(Reshape((2, 2, 512)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # # Conv 1: 4x4x256
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # Conv 2: 8x8x128
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # Conv 3: 16x16x64
        model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # Conv 4: 32x32x3
        model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                                      activation='tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        init = initializers.RandomNormal(stddev=0.02)

        # Discriminator network
        model = Sequential()

        # Conv 1: 16x16x64
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                                 input_shape=(self.img_shape), kernel_initializer=init))
        model.add(LeakyReLU(0.2))

        # Conv 2:
        model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # Conv 3:
        model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # Conv 3:
        model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # FC
        model.add(Flatten())

        # Output
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/dcgan/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
