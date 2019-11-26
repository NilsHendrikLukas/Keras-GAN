from __future__ import print_function, division

from keras import initializers
from keras.datasets import cifar10
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

from mia_attacks.mia_attacks import logan_mia, distance_mia, featuremap_mia


class DCGAN():
    def __init__(self,
                 n_samples=30000,
                 linspace_triplets_logan=(0, 200, 300),
                 log_logan_mia=False,
                 featuremap_mia_epochs=100,
                 log_featuremap_mia=False,
                 linspace_triplets_dist=(1800, 2300, 1000),
                 log_dist_mia=False,
                 load_model=False):

        self.log_logan_mia = log_logan_mia
        self.log_dist_mia = log_dist_mia
        self.featuremap_mia_epochs=featuremap_mia_epochs
        self.log_featuremap_mia=log_featuremap_mia
        self.n_samples = n_samples
        self.linspace_triplets_logan = linspace_triplets_logan
        self.linspace_triplets_dist = linspace_triplets_dist    
        
        self.dataset = 'cifar10'
    
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.train = self.train
            

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

        # Load the dataset
        (self.X_train, _), (X_test, _) = cifar10.load_data()
        # Rescale 0 to 1
        self.X_train = (self.X_train - 127.5) / 127.5

        self.logit_discriminator = None
        self.gan_discriminator = None
        self.featuremap_discriminator = None
        self.featuremap_attacker = None

    def transposed_conv(self, model, out_channels):
        model.add(Conv2DTranspose(out_channels, [5, 5], strides=(
            2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        return model

    def conv(self, model, out_channels):
        model.add(Conv2D(out_channels, (5, 5),
                         kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        return model

    def build_generator(self):

        # Generator network
        downsize = 3
        scale = 2 ** downsize

        model = Sequential()
        model.add(Dense(32 // scale * 32 // scale * 1024,
                        input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(
            Reshape([32 // scale, 32// scale, 1024]))
        model = self.transposed_conv(model, 64)
        if (downsize == 3):
            model = self.transposed_conv(model, 32)
        model.add(Conv2DTranspose(3, [5, 5], strides=(
            2, 2), activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))


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

    def get_gan_discriminator(self):
        if self.gan_discriminator is None:
            feature_maps = self.discriminator.layers[-3].layers[-1].output
            new_logits = Dense(1)(feature_maps)
            self.gan_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[new_logits])
            self.gan_discriminator.name = "gan_discriminator"
        self.gan_discriminator.layers[-1].set_weights(self.discriminator.layers[-2].get_weights())
        return self.gan_discriminator

    def get_logit_discriminator(self):
        if self.logit_discriminator is None:
            feature_maps = self.discriminator.layers[-3].layers[-1].output
            new_logits = Dense(10)(feature_maps)
            self.logit_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[new_logits])
            self.logit_discriminator.name = "logit_discriminator"
        self.logit_discriminator.layers[-1].set_weights(self.discriminator.layers[-1].get_weights())
        return self.logit_discriminator

    def get_featuremap_discriminator(self):
        if self.featuremap_discriminator is None:
            feature_maps = self.discriminator.layers[-3].layers[-1].output
            print("FeatureMaps Layer: {}".format(feature_maps))
            self.featuremap_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[feature_maps])
            self.featuremap_discriminator.name = "featuremap_discriminator"
        return self.featuremap_discriminator

    def train(self, epochs, batch_size=128, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            imgs = self.X_train[idx]

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
            print("{} [D loss {:.2f}, acc.: {:.2f}] [G loss {:.2f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss),
                  end='\r')

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                # Perform the MIA
                self.execute_logan_mia()
                #self.execute_dist_mia()
                self.execute_featuremap_mia()

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
                axs[i,j].imshow(gen_imgs[cnt, :,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/dcgan/images/cifar_%d.png" % epoch)
        plt.close()


    def get_gan_discriminator(self):
            print("GAN discriminator")
            if self.gan_discriminator is None:
                feature_maps = self.discriminator.layers[-1].layers[-1].output
                new_logits = Dense(1)(feature_maps)
                self.gan_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[new_logits])
                self.gan_discriminator.name = "gan_discriminator"
            self.gan_discriminator.layers[-1].set_weights(self.discriminator.layers[-1].get_weights())
            return self.gan_discriminator

    def get_logit_discriminator(self):
        print("Logit discriminator")
        if self.logit_discriminator is None:
            feature_maps = self.discriminator.layers[-1].layers[-2].output
            new_logits = Dense(1)(feature_maps)
            self.logit_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[new_logits])
            self.logit_discriminator.name = "logit_discriminator"
        self.logit_discriminator.layers[-1].set_weights(self.discriminator.layers[-1].layers[-1].get_weights())
        return self.logit_discriminator

    def get_featuremap_discriminator(self):
        print("Featuremap discriminator")
        if self.featuremap_discriminator is None:
            feature_maps = self.discriminator.layers[-1].layers[-2].output
            print("FeatureMaps Layer: {}".format(feature_maps))
            self.featuremap_discriminator = Model(inputs=[self.discriminator.layers[1].get_input_at(0)], outputs=[feature_maps])
            self.featuremap_discriminator.name = "featuremap_discriminator"
        return self.featuremap_discriminator

    def execute_featuremap_mia(self):
        n = 500
        x_in, x_out = self.X_train[0:n], self.X_train[n:self.n_samples + n]
        if self.featuremap_discriminator is None:
            self.featuremap_discriminator = self.get_featuremap_discriminator()
        max_acc = featuremap_mia(self.featuremap_discriminator, None, 10, x_in, x_out)

    def execute_dist_mia(self):
        n = 50
        x_in, x_out = self.X_train[0:n], self.X_train[n:self.n_samples + n]
        max_acc = distance_mia(self.generator, x_in, x_out)

        with open('Keras-GAN/dcgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def execute_logan_mia(self):
        n = 1000
        x_in, x_out = self.X_train[0:n], self.X_train[n:self.n_samples+n]
        max_acc = logan_mia(self.get_logit_discriminator(), x_in, x_out)

        with open('Keras-GAN/dcgan/logs/logan_mia.csv', mode='w+') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/dcgan/images/%d.png" % epoch)
        plt.close()
        return gen_imgs

    def load_model(self):

        def load(model, model_name):
            weights_path = "Keras-GAN/dcgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_weight": weights_path}
            model.load_weights(options['file_weight'])

        load(self.generator, "generator_"+str(self.dataset))
        load(self.discriminator, "discriminator_"+str(self.dataset))

    def save_model(self):

        def save(model, model_name):
            model_path = "Keras-GAN/dcgan/saved_model/%s.json" % model_name
            weights_path = "Keras-GAN/dcgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator_"+str(self.dataset))
        save(self.discriminator, "discriminator_"+str(self.dataset))

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    dcgan = DCGAN()
    
    dcgan.train(2000)
    dcgan.save_model()
    
    # dcgan.load_model()

    # n = 500
    # x_in, x_out = X_train[0:n], X_train[100000:100000+n]
    # x_in = x_in.astype(np.float32)-127.5/127.5
    # x_out = x_out.astype(np.float32) - 127.5 / 127.5

    # dcgan.featuremap_mia(x_in, x_out, plot_graph=True)
    # dcgan.distance_mia(x_in, x_out, plot_graph=True)
    # dcgan.logan_mia(x_in, x_out, plot_graph=True)