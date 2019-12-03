from __future__ import print_function, division

from keras.datasets import mnist
from emnist import extract_training_samples
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
import sys

import numpy as np
from sklearn.utils import shuffle


class DCGAN():
    def __init__(self,
                 max_data=40000,
                 mia_attacks=None,
                 use_advreg=False):
        """
                :param max_data How much x_in data to use
                :param mia_attacks List with following possible values ["logan", "dist", "featuremap"]
                :param use_advreg Build with advreg or without
        """

        if mia_attacks is None:
            mia_attacks = []

        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.use_advreg = use_advreg
        self.mia_attacks = mia_attacks

        np.random.seed(0)

        #######################################
        def normalize(data):
            return np.reshape(data / 127.5 - 1., (-1, *self.img_shape))

        # Load, normalize and split the dataset
        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = normalize(self.x_train)

        self.x_out, y_out = extract_training_samples('digits')
        self.x_out = normalize(self.x_out)

        self.x_train = self.x_train[:max_data]

        print("Loading with {} data samples!".format(len(self.x_train)))

        #########################################

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.featuremap_model, self.discriminator, self.critic_model_with_advreg, self.advreg_model = self.build_discriminator(
            optimizer)

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

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def build_advreg(self, input_shape):
        """ Build the model for the adversarial regularizer
        """
        advreg_in = Input(input_shape)

        l0 = Dense(units=500)(advreg_in)
        l1 = Dropout(0.2)(l0)
        l2 = Dense(units=250)(l1)
        l3 = Dropout(0.2)(l2)
        l4 = Dense(units=10)(l3)

        advreg_out = Dense(units=1, activation="linear")(l4)

        return Model(advreg_in, advreg_out)


    def build_generator(self):

        input_data = Input((self.latent_dim,))

        l0 = Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim)(input_data)
        l1 = Reshape((7, 7, 128))(l0)
        l2 = UpSampling2D()(l1)
        l3 = Conv2D(128, kernel_size=3, padding="same")(l2)
        l4 = BatchNormalization(momentum=0.8)(l3)
        l5 = Activation("relu")(l4)
        l6 = UpSampling2D()(l5)
        l7 = Conv2D(64, kernel_size=3, padding="same")(l6)
        l8 = BatchNormalization(momentum=0.8)(l7)
        l9 = Activation("relu")(l8)
        l10 = Conv2D(self.channels, kernel_size=3, padding="same")(l9)

        output = Activation("tanh")(l10)

        return Model(input_data, output)

    def build_discriminator(self, optimizer):
        dropout = 0.25
        img_shape = (28, 28, 1)

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
        critic_out = Dense(1, activation="sigmoid", name="critic_out")(featuremaps)

        """ Build the critic WITHOUT the adversarial regularization
                        """
        critic_model_without_advreg = Model(inputs=[critic_in], outputs=[critic_out])

        critic_model_without_advreg.compile(optimizer=optimizer,
                                            metrics=["accuracy"],
                                            loss='binary_crossentropy')

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

        def advreg(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred)

        advreg_model.compile(optimizer=optimizer,
                             metrics=["accuracy"],
                             loss=advreg)

        """ Build the critic WITH the adversarial regularization 
        """
        critic_model_with_advreg = Model(inputs=[critic_in], outputs=[critic_out, mia_pred])

        advreg_model.trainable = False

        def critic_out(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred)

        def mia_pred(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred)

        critic_model_with_advreg.compile(optimizer=optimizer,
                                         metrics=["accuracy"],
                                         loss={
                                             "critic_out": critic_out,
                                             "mia_pred": mia_pred
                                         })

        return featuremap_model, critic_model_without_advreg, critic_model_with_advreg, advreg_model

    def train(self, epochs, batch_size=128, save_interval=50):
        logan_precisions, featuremap_precisions = [], []

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, len(self.x_train), batch_size)
            imgs = self.x_train[idx]

            idx_out = np.random.randint(0, len(self.x_out), batch_size)
            imgs_out = self.x_out[idx_out]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            if self.use_advreg:
                # Train the critic to make the advreg model produce FAKE labels
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)  # valid data
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

                # Tried random, tried valid, next try skipping
                self.critic_model_with_advreg.train_on_batch(imgs_out, [valid, valid])

                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            else:
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)


            # ---------------------
            #  Train AdvReg
            #  Do this in the outer loop to give the discriminator a chance to adapt
            # ---------------------
            if self.use_advreg:
                def sample_target(size):
                    return np.random.randint(0, 2, size)

                idx_out = np.random.randint(0, len(self.x_out), batch_size)
                imgs_out = self.x_out[idx_out]

                idx_in = np.random.randint(0, len(self.x_train), batch_size)
                imgs = self.x_train[idx_in]

                adv_x, adv_y = shuffle(np.concatenate((imgs, imgs_out)), np.concatenate((valid, fake)))
                d_loss_advreg = self.advreg_model.train_on_batch(adv_x, adv_y)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # ---------------------
            #  Debug Output
            # ---------------------

            log = ""
            # Compute the "real" epoch (passes through the dataset)
            log = log + "[{}/{}]".format((epoch * batch_size) // len(self.x_train),
                                         (epochs * batch_size) // len(self.x_train))
            if "logan" in self.mia_attacks:
                precision = self.logan_mia(self.discriminator)
                logan_precisions.append(precision)
                log = log + "[LOGAN Prec: {:.3f}]".format(precision)
            if "featuremap" in self.mia_attacks:
                precision = self.featuremap_mia()
                featuremap_precisions.append(precision)
                log = log + "[FM Prec: {:.3f}]".format(precision)

            if self.use_advreg:
                log = log + "[A loss: %f]" % (1 - d_loss_advreg[0])

            log = log + "%d [D loss: %f] [G loss: %f] " % (epoch, 1 - d_loss[0], 1 - g_loss)
            print(log)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # ---------------------
                #  Plot Statistics
                # ---------------------

                for attack in self.mia_attacks:
                    if attack == "logan":
                        plt.plot(np.arange(len(logan_precisions)), logan_precisions, color="blue")
                        plt.hlines(0.5, 0, len(logan_precisions), linestyles="dashed")
                    if attack == "featuremap":
                        plt.plot(np.arange(len(featuremap_precisions)), featuremap_precisions, color="red")
                        plt.hlines(0.5, 0, len(logan_precisions), linestyles="dashed")
                    plt.ylim((0, 1))
                    plt.xlabel("Iterations")
                    plt.ylabel("Success")
                    plt.show()
                self.save_imgs(epoch)

    def featuremap_mia(self, threshold=0.2):
        """
        Takes the classifiers featuremaps and predicts on them
        """
        test_size = 256
        epochs = 5
        batch_size = min(128, len(self.x_train))

        for e in range(epochs):
            idx_in  = np.random.randint(0, len(self.x_train), batch_size)
            idx_out = np.random.randint(0, len(self.x_out), batch_size)

            x_in, x_out = self.x_train[idx_in], self.x_out[idx_out]

            valid = -np.ones((batch_size, 1))
            fake = np.ones((batch_size, 1))
            d_loss_real = self.advreg_model.train_on_batch(x_in, valid)
            d_loss_fake = self.advreg_model.train_on_batch(x_out, fake)

        idx_in = np.random.randint(0, len(self.x_train), test_size)
        idx_out = np.random.randint(0, len(self.x_out), test_size)

        y_preds_in = self.advreg_model.predict(self.x_train[idx_in])
        y_preds_out = self.advreg_model.predict(self.x_out[idx_out])

        # -1 means in, 1 means out
        print("Accuracy In: {}".format(len(np.where(np.sign(y_preds_in) == -1)[0])))
        print("Accuracy Out: {}".format(len(np.where(np.sign(y_preds_out) == 1)[0])))

        """
            True negatives
        """
        p = np.concatenate((y_preds_in, y_preds_out)).flatten().argsort()
        p = p[-int((len(y_preds_out) + len(y_preds_in)) * threshold):]

        # How many of the ones that are in are covered:
        true_negatives, = np.where(p >= len(y_preds_in))
        false_negatives, = np.where(p < len(y_preds_in))

        print("True Negatives: {}/{}".format(len(true_negatives), len(p)))
        print("False Negatives: {}".format(len(false_negatives)))

        precision = len(true_negatives) / (len(true_negatives) + len(false_negatives))

        """
            True Positives
        """
        p = np.concatenate((y_preds_in, y_preds_out)).flatten().argsort()
        p = p[:int((len(y_preds_out) + len(y_preds_in)) * threshold)]

        # How many of the ones that are in are covered:
        true_positives, = np.where(p < len(y_preds_in))
        false_positives, = np.where(p >= len(y_preds_in))

        print("True Positives: {}/{}".format(len(true_positives), len(p)))
        print("False Positives: {}".format(len(false_positives)))

        accuracy = (len(true_positives)+len(true_negatives)) / (len(true_positives)+len(true_negatives)+len(false_positives)+len(false_negatives))

        return accuracy


    def logan_mia(self,
                  critic_model,
                  threshold=0.2):
        """
        LOGAN is an attack that passes all examples through the critic and classifies those as members with
        a threshold higher than the passed value
        """
        batch_size = min(1024, len(self.x_train))
        idx_in, idx_out = np.random.randint(0, len(self.x_train), batch_size), np.random.randint(0, len(self.x_out), batch_size)
        x_in, x_out = self.x_train[idx_in], self.x_out[idx_out]

        y_preds_in = critic_model.predict(x_in)
        y_preds_out = critic_model.predict(x_out)

        # Get 10% with highest confidence
        p = np.abs(np.concatenate((y_preds_in, y_preds_out))).flatten().argsort()

        print("In: {}, Out: {}".format(np.mean(y_preds_in), np.mean(y_preds_out)))

        p = p[-int((len(y_preds_out)+len(y_preds_in))*threshold):]

        # How many of the ones that are in are covered:
        false_positives, = np.where(p >= len(y_preds_in))
        true_positives, = np.where(p < len(y_preds_in))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))

        return precision

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
        plt.show()
        try:
            fig.savefig("dcgan/images/mnist_%d.png" % epoch)
        except:
            pass
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN(use_advreg=True, mia_attacks=["logan", "featuremap"])
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)