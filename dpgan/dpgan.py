from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras_gradient_noise import add_gradient_noise
from mnist_models import build_generator, build_critic
from emnist import extract_training_samples

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class DPGAN():
    def __init__(self,
                 max_data=40000):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        def normalize(data):
            return np.reshape((data.astype(np.float32) - 127.5) / 127.5, (-1, *self.img_shape))
        # Load, normalize and split the dataset
        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = normalize(self.x_train)

        self.x_out, y_out = extract_training_samples('digits')
        self.x_out = normalize(self.x_out)

        self.x_train = self.x_train[:max_data]

        print("Loading with {} data samples!".format(len(self.x_train)))

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        NoisyRMSprop = add_gradient_noise(RMSprop)
        noisyoptimizer = NoisyRMSprop(lr=0.00005, noise_eta=0.3, noise_gamma=0.55)
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic(optimizer)

        # Build the generator
        self.generator = build_generator()

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

    def build_critic(self, optimizer):
        """ Build the discriminators for MNIST with advreg
        """
        img_shape = (28, 28, 1)

        dropout = 0 # 0.25

        critic_in = Input(img_shape)

        l0 = Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(critic_in)
        l1 = LeakyReLU(alpha=0.2)(l0)
        l2 = Dropout(dropout)(l1)
        l3 = Conv2D(32, kernel_size=3, strides=2, padding="same")(l2)
        l4 = ZeroPadding2D(padding=((0, 1), (0, 1)))(l3)
        l5 = BatchNormalization(momentum=0.8)(l4)
        l6 = LeakyReLU(alpha=0.2)(l5)
        l7 = Dropout(dropout)(l6)
        l8 = Conv2D(64, kernel_size=3, strides=2, padding="same")(l7)
        l9 = BatchNormalization(momentum=0.8)(l8)
        l10 = LeakyReLU(alpha=0.2)(l9)
        l11 = Dropout(dropout)(l10)
        l12 = Conv2D(128, kernel_size=3, strides=1, padding="same")(l11)
        l13 = BatchNormalization(momentum=0.8)(l12)
        l14 = LeakyReLU(alpha=0.2)(l13)
        l15 = Dropout(dropout)(l14)
        featuremaps = Flatten()(l15)
        critic_out = Dense(1, name="critic_out")(featuremaps)

        """ Build the critic WITHOUT the adversarial regularization
                """
        critic_model_without_advreg = Model(inputs=[critic_in], outputs=[critic_out])

        critic_model_without_advreg.compile(optimizer=optimizer,
                                            metrics=["accuracy"],
                                            loss=self.wasserstein_loss)

        return critic_model_without_advreg

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, self.x_train.shape[0], batch_size)
                imgs = self.x_train[idx]
                
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
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

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
                #axs[i, j].imshow(gen_imgs[cnt, :, :].squeeze())
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        try:
            fig.savefig("Keras-GAN/dpgan/images/%d.png" % (epoch))
        except:
            pass

        plt.close()

        return gen_imgs


if __name__ == '__main__':
    dpgan = DPGAN()
    dpgan.train(epochs=4000, batch_size=32, sample_interval=5)
