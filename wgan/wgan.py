from __future__ import print_function, division

import sys

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Lambda, Conv2D, LeakyReLU, ZeroPadding2D, BatchNormalization, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from sklearn.utils import shuffle

from mnist_models import build_generator

from mia_attacks.mia_attacks import distance_mia, featuremap_mia, logan_top_n

class WGAN():
    def __init__(self,
                 mia_attacks=None,
                 use_advreg=True):
        """
        :param mia_attacks List with following possible values ["logan", "dist", "featuremap"]
        :param use_advreg Build with advreg or without

        """
        if mia_attacks is None:
            mia_attacks = []

        # Define input shapes
        self.img_shape = (28, 28, 1)
        self.latent_dim = 100
        self.use_advreg = use_advreg
        self.mia_attacks = mia_attacks

        n_out = 10000  # 10K samples are out!

        np.random.seed(0)
        #######################################
        # Load, normalize and split the dataset
        (self.x_train, _), (_, _) = mnist.load_data()
        print(type(self.x_train))
        self.x_train = np.reshape(self.x_train / 255, (-1, *self.img_shape))

        self.x_out = self.x_train[:n_out]  # 10K samples are out!
        self.x_train = self.x_train[n_out:]

        #########################################
        # Build and compile the critic and generator
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Gets all (compiled!) critic models
        self.featuremap_model, self.critic_model, self.critic_model_with_advreg, self.advreg_model = self.build_critic()

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic_model.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic_model(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def adv_reg_loss(self, y_true, y_pred):
        alpha = 0.9

        # priv_diff = inference - K.ones(K.shape(inference))
        priv_diff = y_true - y_pred
        privacy_loss = K.pow(priv_diff, 2)

        return -alpha * K.mean(privacy_loss)

        # return -K.mean(y_true * y_pred)

    def adv_reg_loss_neg(self, y_true, y_pred):
        return - self.adv_reg_loss(y_true, y_pred)

    def build_advreg(self, input_shape):
        """ Build the model for the adversarial regularizer
        """
        advreg_in = Input(input_shape)

        l0 = Dense(units=500)(advreg_in)
        l1 = Dropout(0.2)(l0)
        l2 = Dense(units=250)(l1)
        l3 = Dropout(0.2)(l2)
        l4 = Dense(units=10)(l3)

        advreg_out = Dense(units=1, activation="sigmoid")(l4)

        return Model(advreg_in, advreg_out)

    def build_generator(self):
        return build_generator()

    def build_critic(self):
        """ Build the discriminators for MNIST with advreg
        """
        img_shape = (28, 28, 1)

        critic_in = Input(img_shape)

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

        """ Build the adversarial regularizer
        If no adversarial regularization is required, disable it in the training function /!\
        """
        featuremap_model = Model(inputs=[critic_in], outputs=[featuremaps])

        advreg = self.build_advreg(input_shape=(2048,))
        mia_pred = advreg(featuremap_model(critic_in))

        naming_layer = Lambda(lambda x: x, name='mia_pred')
        mia_pred = naming_layer(mia_pred)

        advreg_model = Model(inputs=[critic_in], outputs=[mia_pred])

        # Do not train the critic when updating the adversarial regularizer
        featuremap_model.trainable = False

        advreg_model.compile(optimizer=Adam(1e-3),
                       metrics=["accuracy"],
                       loss=self.wasserstein_loss)

        """ Build the critic WITH the adversarial regularization 
        """
        critic_model_with_advreg = Model(inputs=[critic_in], outputs=[critic_out, mia_pred])

        advreg_model.trainable = False

        critic_model_with_advreg.compile(optimizer=Adam(1e-3),
                       metrics=["accuracy"],
                       loss={
                           "critic_out": self.wasserstein_loss,
                           "mia_pred": self.wasserstein_loss
                       })

        """ Build the critic WITHOUT the adversarial regularization
        """
        critic_model_without_advreg = Model(inputs=[critic_in], outputs=[critic_out])

        critic_model_without_advreg.compile(optimizer=Adam(1e-3),
                                         metrics=["accuracy"],
                                         loss=self.wasserstein_loss)

        return featuremap_model, critic_model_without_advreg, critic_model_with_advreg, advreg_model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(self.n_critic):
                # Select a random batch of images
                idx = np.random.randint(0, len(self.x_train), batch_size)
                imgs = self.x_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                """ Train the critic
                NOTE: A critic WITH advreg has 2 outputs (valid, mia_pred), so we need to distinguish 
                If advreg is used, we have to pass x_in and x_out after training the discriminator 
                """
                if self.use_advreg:
                    # First train the critic
                    d_loss_real = self.critic_model_with_advreg.train_on_batch(imgs, [valid, valid])    # valid data, valid (is in the dataset)
                    d_loss_fake = self.critic_model_with_advreg.train_on_batch(gen_imgs, [fake, valid]) # fake data, valid (is in the dataset)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # Now update the advreg
                    idx_out = np.random.randint(0, len(self.x_out), batch_size)
                    imgs_out = self.x_out[idx_out]

                    adv_x, adv_y = shuffle(np.concatenate((imgs, imgs_out)), np.concatenate((valid, fake)))
                    d_loss_advreg = self.advreg_model.train_on_batch(adv_x, adv_y)
                else:
                    d_loss_real = self.critic_model.train_on_batch(imgs, valid)
                    d_loss_fake = self.critic_model.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic_model.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

                # Perform the MIA
                #### Added
                def overfit_discriminator(epochs):
                    for i in range(epochs):
                        idx = np.random.randint(0, self.x_train.shape[0], batch_size)
                        imgs = self.x_train[idx]
                        self.critic_model.train_on_batch(imgs, valid)

                overfit_discriminator(0)
                #### Added

                self.execute_logan_mia(self.critic_model)
                # self.execute_dist_mia()
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
                #axs[i, j].imshow(gen_imgs[cnt, :, :].squeeze())
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/wgan/images/%d.png" % (epoch))
        plt.close()

        return gen_imgs

    def get_gan_discriminator(self):
        print("GAN critic")
        if self.gan_discriminator is None:
            feature_maps = self.critic_model.layers[-1].layers[-1].output
            new_logits = Dense(1)(feature_maps)
            self.gan_discriminator = Model(inputs=[self.critic_model.layers[1].get_input_at(0)], outputs=[new_logits])
            self.gan_discriminator.name = "gan_discriminator"
        self.gan_discriminator.layers[-1].set_weights(self.critic_model.layers[-1].get_weights())
        return self.gan_discriminator

    def execute_featuremap_mia(self):
        n = 10000
        n_val = 500  # Samples used only in validation
        val_in, val_out = self.x_train[:n_val], \
                          self.x_train[self.n_samples:self.n_samples + n_val]

        train_in = self.x_train[n_val:self.n_samples + n_val]
        train_out = self.x_train[self.n_samples + n_val:self.n_samples + n_val + len(train_in)]
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
        x_in, x_out = self.x_train[0:n], self.x_train[self.n_samples:self.n_samples + n]

        max_acc = distance_mia(self.generator, x_in, x_out)

        with open('Keras-GAN/dcgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def execute_logan_mia(self, critic_model):
        n = 1000
        n_val = 500  # Samples used ONLY in validation
        val_in, val_out = self.x_train[:n_val], \
                          self.X_test[:n_val]

        train_in = self.x_train[n_val:n_val + n_val]
        train_out = self.X_test[n_val:n_val + len(train_in)]
        train_in, train_out = shuffle(train_in, train_out)

        # max_acc = logan_mia(self.get_logit_discriminator(critic_model), train_in, train_out)
        max_acc = logan_top_n(self.get_featuremap_output(critic_model), train_in, train_out, n_val // 10)

        with open('Keras-GAN/dcgan/logs/logan_mia.csv', mode='w+') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def load_model(self):

        def load(model, model_name):
            weights_path = "Keras-GAN/wgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_weight": weights_path}
            model.load_weights(options['file_weight'])

        load(self.generator, "generator_" + str(self.dataset))
        load(self.critic_model, "discriminator_" + str(self.dataset))

    def save_model(self):

        def save(model, model_name):
            model_path = "Keras-GAN/wgan/saved_model/%s.json" % model_name
            weights_path = "Keras-GAN/wgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator_" + str(self.dataset))
        save(self.critic_model, "discriminator_" + str(self.dataset))


if __name__ == '__main__':
    wgan = WGAN()

    # wgan.train(epochs=4000, batch_size=32, sample_interval=5)
    wgan.train(epochs=40000, batch_size=32, sample_interval=5)
    # wgan.save_model()

    # wgan.load_model()
    # n = 500
    # x_in, x_out = X_train[0:n], X_train[100000:100000+n]
    # x_in = x_in.astype(np.float32)-127.5/127.5
    # x_out = x_out.astype(np.float32) - 127.5 / 127.5
    # wgan.featuremap_mia(x_in, x_out, plot_graph=True)
    # wgan.distance_mia(x_in, x_out, plot_graph=True)
    # wgan.logan_mia(x_in, x_out, plot_graph=True)
