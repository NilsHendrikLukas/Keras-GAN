from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self,
                 input_shape=(108, 108, 3)):
        # Input shape
        self.img_rows, self.img_cols, self.channels = input_shape
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
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

        input_noise = Input(shape=(self.latent_dim,))
        d = Dense(1024, activation="relu")(input_noise)
        d = Dense(1024, activation="relu")(input_noise)
        d = Dense(128 * 8 * 8, activation="relu")(d)
        d = Reshape((8, 8, 128))(d)

        d = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
        d = Conv2D(64, (1, 1), activation='relu', padding='same', name="block_4")(d)  ## 16,16

        d = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
        d = Conv2D(64, (1, 1), activation='relu', padding='same', name="block_5")(d)  ## 32,32

        if self.input_shape[0] == 64:
            d = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
            d = Conv2D(64, (1, 1), activation='relu', padding='same', name="block_6")(d)  ## 64,64

        img = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name="final_block")(d)  ## 32, 32
        model = Model(input_noise, img)
        return model

    def build_discriminator(self):

        input_img = Input(shape=self.input_shape)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(input_img, out)

        return model

    def train(self,
              x_train,
              epochs,
              batch_size=128,
              save_interval=50):

        # Rescale -1 to 1
        x_train = x_train / np.mean(x_train) - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

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
        fig.savefig("Keras-GAN/dcgan/images/celeba_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)