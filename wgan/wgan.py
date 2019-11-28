from __future__ import print_function, division

from keras import initializers
from keras.datasets import cifar10, mnist
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import keras.backend as K

import sys

import numpy as np
from sklearn.utils import shuffle

import sys
sys.path.append('Keras-GAN')
from mia_attacks.mia_attacks import logan_mia, distance_mia, featuremap_mia
class WGAN():
    def __init__(self,
                 n_samples=5000,
                 linspace_triplets_logan=(0, 200, 300),
                 log_logan_mia=False,
                 featuremap_mia_epochs=100,
                 log_featuremap_mia=False,
                 linspace_triplets_dist=(1800, 2300, 1000),
                 log_dist_mia=False,
                 load_model=False,
                 dataset='mnist'):

        # Input shape
        # CIFAR
        # self.img_rows = 32
        # self.img_cols = 32
        # self.channels = 3
        # MNIST
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.log_logan_mia = log_logan_mia
        self.log_dist_mia = log_dist_mia
        self.featuremap_mia_epochs=featuremap_mia_epochs
        self.log_featuremap_mia=log_featuremap_mia
        self.n_samples = n_samples
        self.linspace_triplets_logan = linspace_triplets_logan
        self.linspace_triplets_dist = linspace_triplets_dist   

        self.dataset = 'mnist'

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


        # Load the dataset
        # (self.X_train, _), (X_test, _) = cifar10.load_data()
        (self.X_train, _), (X_test, _) = mnist.load_data()
        # Rescale 0 to 1
        self.X_train = (self.X_train - 127.5) / 127.5

        #MNIST only
        self.X_train = np.expand_dims(self.X_train, axis=3)

        self.logit_discriminator = None
        self.gan_discriminator = None
        self.featuremap_discriminator = None
        self.featuremap_attacker = None


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

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
                idx = np.random.randint(0, self.n_samples, batch_size)
                imgs = self.X_train[idx]
                
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
            print("{} [D loss {:.2f}, acc.: {:.2f}] [G loss {:.2f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss),
                  end='\r')
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                # Perform the MIA
                #### Added
                def overfit_discriminator(epochs):
                    for i in range(epochs):
                        idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                        imgs = self.X_train[idx]
                        self.critic.train_on_batch(imgs, valid)

                overfit_discriminator(0)
                #### Added

                self.execute_logan_mia()
                #self.execute_dist_mia()
                # self.execute_featuremap_mia()

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
                # axs[i,j].imshow(gen_imgs[cnt, :,:])
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/wgan/images/mnist_%d.png" % epoch)
        plt.close()

        return gen_imgs

    def get_gan_discriminator(self):
        print("GAN critic")
        if self.gan_discriminator is None:
            feature_maps = self.critic.layers[-1].layers[-1].output
            new_logits = Dense(1)(feature_maps)
            self.gan_discriminator = Model(inputs=[self.critic.layers[1].get_input_at(0)], outputs=[new_logits])
            self.gan_discriminator.name = "gan_discriminator"
        self.gan_discriminator.layers[-1].set_weights(self.critic.layers[-1].get_weights())
        return self.gan_discriminator

    def get_logit_discriminator(self):
        print("Logit discriminator")
        if self.logit_discriminator is None:
            feature_maps = self.critic.layers[-1].layers[-2].output
            new_logits = Dense(1)(feature_maps)
            self.logit_discriminator = Model(inputs=[self.critic.layers[1].get_input_at(0)], outputs=[new_logits])
            self.logit_discriminator.name = "logit_discriminator"
        self.logit_discriminator.layers[-1].set_weights(self.critic.layers[-1].layers[-1].get_weights())
        return self.logit_discriminator

    def get_featuremap_discriminator(self):
        print("Featuremap discriminator")
        if self.featuremap_discriminator is None:
            feature_maps = self.critic.layers[-1].layers[-2].output
            print("FeatureMaps Layer: {}".format(feature_maps))
            self.featuremap_discriminator = Model(inputs=[self.critic.layers[1].get_input_at(0)], outputs=[feature_maps])
            self.featuremap_discriminator.name = "featuremap_discriminator"
        return self.featuremap_discriminator


    def execute_featuremap_mia(self):
        n = 10000
        n_val = 500  # Samples used only in validation
        val_in, val_out = self.X_train[:n_val], \
                          self.X_train[self.n_samples:self.n_samples + n_val]

        train_in = self.X_train[n_val:self.n_samples+n_val]
        train_out = self.X_train[self.n_samples + n_val:self.n_samples + n_val + len(train_in)]
        train_in, train_out = shuffle(train_in, train_out)
        train_in, train_out = train_in[:n], train_out[:n]

        if self.featuremap_discriminator is None:
            self.featuremap_discriminator = self.get_featuremap_discriminator()
        self.featuremap_attacker = featuremap_mia(self.featuremap_discriminator,
                                                  self.featuremap_attacker,
                                                  epochs=25,
                                                  x_in=train_in,
                                                  x_out=train_out,
                                                  val_in=val_in,
                                                  val_out=val_out)

    def execute_dist_mia(self):
        n = 50
        x_in, x_out = self.X_train[0:n], self.X_train[self.n_samples:self.n_samples + n]

        max_acc = distance_mia(self.generator, x_in, x_out)

        with open('Keras-GAN/dcgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def execute_logan_mia(self):
        n = 500
        n_val = 500     # Samples used ONLY in validation
        val_in, val_out = self.X_train[:n_val], \
                          self.X_train[self.n_samples:self.n_samples+n_val]

        train_in = self.X_train[n_val:self.n_samples+n_val]
        train_out = self.X_train[self.n_samples+n_val:self.n_samples+n_val+len(train_in)]
        train_in, train_out = shuffle(train_in, train_out)
        train_in, train_out = train_in[:n], train_out[:n]

        max_acc = logan_mia(self.get_logit_discriminator(), train_in, train_out)

        with open('Keras-GAN/dcgan/logs/logan_mia.csv', mode='w+') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def load_model(self):

        def load(model, model_name):
            weights_path = "Keras-GAN/wgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_weight": weights_path}
            model.load_weights(options['file_weight'])

        load(self.generator, "generator_"+str(self.dataset))
        load(self.critic, "discriminator_"+str(self.dataset))

    def save_model(self):

        def save(model, model_name):
            model_path = "Keras-GAN/wgan/saved_model/%s.json" % model_name
            weights_path = "Keras-GAN/wgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator_"+str(self.dataset))
        save(self.critic, "discriminator_"+str(self.dataset))

if __name__ == '__main__':
    wgan = WGAN()

    wgan.train(epochs=4000, batch_size=32, sample_interval=5)
    # wgan.train(epochs=40000, batch_size=32, sample_interval=50)
    wgan.save_model()

    # wgan.load_model()
    # n = 500
    # x_in, x_out = X_train[0:n], X_train[100000:100000+n]
    # x_in = x_in.astype(np.float32)-127.5/127.5
    # x_out = x_out.astype(np.float32) - 127.5 / 127.5
    # wgan.featuremap_mia(x_in, x_out, plot_graph=True)
    # wgan.distance_mia(x_in, x_out, plot_graph=True)
    # wgan.logan_mia(x_in, x_out, plot_graph=True)