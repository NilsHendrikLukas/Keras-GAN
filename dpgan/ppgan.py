import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input, Lambda
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import initializers
from emnist import extract_training_samples

from gradient_noise import add_gradient_noise

class PPGAN():
    def __init__(self,
                 max_data=60000,
                 eps = 50,
                 gamma = 0.000001,
                 mia_attacks=None
                 ):
        self.mia_attacks = mia_attacks
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        NoisyAdam = add_gradient_noise(Adam)

        K.image_data_format()

        noise_std = 2*128/max_data*np.sqrt(np.log(1/gamma))/eps

        print("Setting noise stadard deviation to " + str(noise_std))
        np.random.seed(0) # Deterministic output.
        self.random_dim = 100 # For consistency with other GAN implementations.

        def normalize(data):
            return np.reshape((data.astype(np.float32) - 127.5) / 127.5, (-1, *self.img_shape))
        # Load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = normalize(X_train)
        self.X_train = X_train.reshape(60000, 784)

        self.x_out, y_out = extract_training_samples('digits')
        self.x_out = normalize(self.x_out)
        self.x_out = self.x_out.reshape(240000, 784)

        self.X_train = self.X_train[:max_data]
        self.img_shape = self.X_train.shape[1]

        # Generator
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(self.X_train.shape[1], activation='tanh'))

        generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        generator.compile(optimizer=generator_optimizer, loss='binary_crossentropy')
        self.generator = generator

        # Discriminator
        # discriminator = Sequential()
        # discriminator.add(Dense(1024, input_dim=self.X_train.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(512))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(256))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(1, activation='sigmoid'))
        dropout = 0.3
        critic_in = Input((self.img_shape,))
        l0 = Dense(1024, input_shape=(self.img_shape,), kernel_initializer=initializers.RandomNormal(stddev=0.02))(critic_in)
        l1 = LeakyReLU(alpha=0.2)(l0)
        l2 = Dropout(dropout)(l1)
        l3 = Dense(512)(l2)
        l4 = LeakyReLU(alpha=0.2)(l3)
        l5 = Dropout(dropout)(l4)
        l6 = Dense(256)(l5)
        l7 = LeakyReLU(alpha=0.2)(l6)
        featuremaps = Dropout(dropout)(l7)
        critic_out = Dense(1, name="critic_out", activation='sigmoid')(featuremaps)
        discriminator = Model(inputs=[critic_in], outputs=[critic_out])

        clipnorm = 5.0
        discriminator_optimizer = NoisyAdam(lr=0.0002, beta_1=0.5, clipnorm=clipnorm, standard_deviation=noise_std)
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        self.discriminator = discriminator

        featuremap_model = Model(inputs=[critic_in], outputs=[featuremaps])


        advreg = self.build_advreg(input_shape=(256,))
        mia_pred = advreg(featuremap_model(critic_in))

        naming_layer = Lambda(lambda x: x, name='mia_pred')
        mia_pred = naming_layer(mia_pred)

        advreg_model = Model(inputs=[critic_in], outputs=[mia_pred])

        # Do not train the critic when updating the adversarial regularizer
        featuremap_model.trainable = False

        advreg_optimizer = Adam(lr=0.0002, beta_1=0.5)
        advreg_model.compile(optimizer=advreg_optimizer,
                       metrics=["accuracy"],
                       loss='binary_crossentropy')
        self.advreg_model = advreg_model

        # GAN
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.random_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)

        gan_optimizer = Adam(lr=0.0002, beta_1=0.5)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
        self.gan = gan

        # Losses for plotting
        self.discriminator_losses = []
        self.generator_losses = []

        self.logan_precisions = []
        self.featuremap_precisions = []

    def build_advreg(self, input_shape):
        """ Build the model for the adversarial regularizer
        """
        advreg_in = Input(input_shape)

        l0 = Dense(units=256)(advreg_in)
        l1 = Dropout(0.2)(l0)
        l2 = Dense(units=128)(l1)
        l3 = Dropout(0.2)(l2)
        l4 = Dense(units=10)(l3)

        advreg_out = Dense(units=1, activation="linear")(l4)

        return Model(advreg_in, advreg_out)

    def plot_loss(self, epoch):
        plt.figure(figsize=(10, 8))
        plt.plot(self.discriminator_losses, label='Discriminitive Loss')
        plt.plot(self.generator_losses, label='Generative Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/gan_loss_epoch_{}.png'.format(epoch))

    def plot_generated_images(self, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, self.random_dim])
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('Keras-GAN/dpgan/images/gan_generated_image_epoch_{}.png'.format(epoch))

    def save_models(self, epoch):
        self.generator.save('Keras-GAN/dpgan/saved_model/gan_generator_epoch_{}.h5'.format(epoch))
        self.discriminator.save('Keras-GAN/dpgan/saved_model/gan_discriminator_epoch_{}.h5'.format(epoch))

    def train(self, epochs=1, batch_size=64):
        batch_count = int(self.X_train.shape[0] / batch_size)

        for e in range(1, epochs+1):
            print('-' * 15, 'Epoch {}'.format(e), '-' * 15)
            for _ in tqdm(range(batch_count)):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                image_batch = self.X_train[np.random.randint(0, self.X_train.shape[0], size=batch_size)]

                # Generate fake MNIST images
                generated_images = self.generator.predict(noise)
                # print np.shape(image_batch), np.shape(generated_images)
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                y_dis = np.zeros(2 * batch_size)
                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.discriminator.trainable = True
                discriminator_loss = self.discriminator.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                y_gen = np.ones(batch_size)
                self.discriminator.trainable = False
                generator_loss = self.gan.train_on_batch(noise, y_gen)

                # ---------------------
                #  Debug Output
                # ---------------------

                log = ""
                # Compute the "real" epoch (passes through the dataset)
                log = log + "[{}/{}]".format((e*batch_size)//len(self.X_train), (e*batch_size)//len(self.X_train))
                if "logan" in self.mia_attacks:
                    precision = self.logan_mia(self.discriminator)
                    self.logan_precisions.append(precision)
                    log = log + "[LOGAN Prec: {:.3f}]".format(precision)
                if "featuremap" in self.mia_attacks:
                    precision = self.featuremap_mia()
                    self.featuremap_precisions.append(precision)
                    log = log + "[FM Prec: {:.3f}]".format(precision)

                log = log + "%d [D loss: %f] [G loss: %f] " % (e, 1 - discriminator_loss, 1 - generator_loss)
                print(log)

            # Store loss of most recent batch from this epoch
            self.discriminator_losses.append(discriminator_loss)
            self.generator_losses.append(generator_loss)

            # ---------------------
            #  Plot Statistics
            # ---------------------

            for attack in self.mia_attacks:
                plt.figure()
                if attack == "logan":
                    plt.plot(np.arange(len(self.logan_precisions)), self.logan_precisions, color="blue")
                    plt.hlines(0.5, 0, len(self.logan_precisions), linestyles="dashed")
                if attack =="featuremap":
                    plt.plot(np.arange(len(self.featuremap_precisions)), self.featuremap_precisions, color="red")
                    plt.hlines(0.5, 0, len(self.featuremap_precisions), linestyles="dashed")
                plt.ylim((0, 1))
                plt.xlabel("Iterations")
                plt.ylabel("Success")
                plt.show()

            self.plot_generated_images(e)
            if e == 1 or e % 20 == 0:
                self.save_models(e)

        # Plot losses from every epoch
        self.plot_loss(e)

    def featuremap_mia(self, threshold=0.2):
        """
        Takes the classifiers featuremaps and predicts on them
        """
        test_size = 256
        epochs = 5
        batch_size = min(128, len(self.X_train))

        for e in range(epochs):
            idx_in  = np.random.randint(0, len(self.X_train), batch_size)
            idx_out = np.random.randint(0, len(self.x_out), batch_size)

            x_in, x_out = self.X_train[idx_in], self.x_out[idx_out]

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            d_loss_real = self.advreg_model.train_on_batch(x_in, valid)
            d_loss_fake = self.advreg_model.train_on_batch(x_out, fake)

        idx_in = np.random.randint(0, len(self.X_train), test_size)
        idx_out = np.random.randint(0, len(self.x_out), test_size)

        y_preds_in = self.advreg_model.predict(self.X_train[idx_in])
        y_preds_out = self.advreg_model.predict(self.x_out[idx_out])

        # 1 means in, 0 means out
        print("Accuracy In: {}".format(len(np.where(np.sign(y_preds_in) == 1)[0])))
        print("Accuracy Out: {}".format(len(np.where(np.sign(y_preds_out) == 0)[0])))

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
        batch_size = min(1024, len(self.X_train))
        idx_in, idx_out = np.random.randint(0, len(self.X_train), batch_size), np.random.randint(0, len(self.x_out), batch_size)
        x_in, x_out = self.X_train[idx_in], self.x_out[idx_out]

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


if __name__ == '__main__':
    ppgan = PPGAN(eps = 50, gamma = 0.000001, mia_attacks=["logan", "featuremap"])
    ppgan.train(100, 128)

