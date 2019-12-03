from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras import initializers
from gradient_noise import add_gradient_noise
from mnist_models import build_generator, build_critic
from emnist import extract_training_samples

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class DPGAN():
    def __init__(self,
                 max_data=40000,
                 noise_std=0.001,
                 mia_attacks=None
                 ):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.mia_attacks = mia_attacks

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
        self.clip_value = 5.0
        NoisyAdam = add_gradient_noise(Adam)
        discriminator_optimizer = NoisyAdam(lr=0.0002, beta_1=0.5, clipnorm=self.clip_value, standard_deviation=noise_std)
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic, self.advreg_model = self.build_critic(discriminator_optimizer, optimizer)

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
        self.combined.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

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
        random_dim = 100
        generator = Sequential()
        generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(28*28, activation='tanh'))
        generator.add(Reshape((28, 28, 1), input_shape=(28*28,)))


        generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        generator.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

        return generator

    def build_critic(self, critic_optimizer, advreg_optimizer):
        """ Build the discriminators for MNIST with advreg
        """
        img_shape = (28, 28, 1)

        dropout = 0.3 # 0.25

        critic_in = Input(img_shape)

        l0 = Dense(1024, input_shape=img_shape, kernel_initializer=initializers.RandomNormal(stddev=0.02))(critic_in)
        l1 = LeakyReLU(alpha=0.2)(l0)
        l2 = Dropout(dropout)(l1)
        l3 = Dense(512)(l2)
        l4 = LeakyReLU(alpha=0.2)(l3)
        l5 = Dropout(dropout)(l4)
        l6 = Dense(256)(l5)
        l7 = LeakyReLU(alpha=0.2)(l6)
        l8 = Dropout(dropout)(l7)
        featuremaps = Flatten()(l8)
        critic_out = Dense(1, name="critic_out", activation='sigmoid')(featuremaps)

        """ Build the critic WITHOUT the adversarial regularization
                """
        critic_model_without_advreg = Model(inputs=[critic_in], outputs=[critic_out])

        critic_model_without_advreg.compile(optimizer=critic_optimizer,
                                            metrics=["accuracy"],
                                            loss='binary_crossentropy')


        """ Build the adversarial regularizer
        If no adversarial regularization is required, disable it in the training function /!\
        """
        featuremap_model = Model(inputs=[critic_in], outputs=[featuremaps])


        advreg = self.build_advreg(input_shape=(256,))
        mia_pred = advreg(featuremap_model(critic_in))

        naming_layer = Lambda(lambda x: x, name='mia_pred')
        mia_pred = naming_layer(mia_pred)

        advreg_model = Model(inputs=[critic_in], outputs=[mia_pred])

        # Do not train the critic when updating the adversarial regularizer
        featuremap_model.trainable = False

        advreg_model.compile(optimizer=advreg_optimizer,
                       metrics=["accuracy"],
                       loss=self.wasserstein_loss)

        return critic_model_without_advreg, advreg_model

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Store all precision values
        logan_precisions, featuremap_precisions = [], []

        # Adversarial ground truths
        soft_valid = np.ones((batch_size, 1))*0.9
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

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
            self.critic.treinable = True
            d_loss_real = self.critic.train_on_batch(imgs, soft_valid)
            d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # ---------------------
            #  Train Generator
            # ---------------------
            self.critic.treinable = False
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            # ---------------------
            #  Debug Output
            # ---------------------

            log = ""
            # Compute the "real" epoch (passes through the dataset)
            log = log + "[{}/{}]".format((epoch*batch_size)//len(self.x_train), (epochs*batch_size)//len(self.x_train))
            if "logan" in self.mia_attacks:
                precision = self.logan_mia(self.critic)
                logan_precisions.append(precision)
                log = log + "[LOGAN Prec: {:.3f}]".format(precision)
            if "featuremap" in self.mia_attacks:
                precision = self.featuremap_mia()
                featuremap_precisions.append(precision)
                log = log + "[FM Prec: {:.3f}]".format(precision)

            log = log + "%d [D loss: %f] [G loss: %f] " % (epoch, 1 - d_loss[0], 1 - g_loss[0])
            print(log)

            # If at save interval => save generated image samples
            if epoch != 0 and epoch % sample_interval == 0:
                # ---------------------
                #  Plot Statistics
                # ---------------------

                for attack in self.mia_attacks:
                    if attack == "logan":
                        plt.plot(np.arange(len(logan_precisions)), logan_precisions, color="blue")
                        plt.hlines(0.5, 0, len(logan_precisions), linestyles="dashed")
                    if attack =="featuremap":
                        plt.plot(np.arange(len(featuremap_precisions)), featuremap_precisions, color="red")
                        plt.hlines(0.5, 0, len(logan_precisions), linestyles="dashed")
                    plt.ylim((0, 1))
                    plt.xlabel("Iterations")
                    plt.ylabel("Success")
                    plt.show()
                # ---------------------
                #  Save images
                # ---------------------
                self.sample_images(epoch)

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

    def execute_dist_mia(self):
        n = 50
        x_in, x_out = self.x_train[0:n], self.x_train[self.n_samples:self.n_samples + n]

        max_acc = distance_mia(self.generator, x_in, x_out)

        with open('Keras-GAN/dcgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

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

    def load_model(self):

        def load(model, model_name):
            weights_path = "Keras-GAN/wgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_weight": weights_path}
            model.load_weights(options['file_weight'])

        load(self.generator, "generator_" + str(self.dataset))
        load(self.critic, "discriminator_" + str(self.dataset))

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
        save(self.critic, "discriminator_" + str(self.dataset))

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
    dpgan = DPGAN(noise_std = 0.0001, mia_attacks=["featuremap", "logan"])
    dpgan.train(epochs=4000, batch_size=32, sample_interval=5)
