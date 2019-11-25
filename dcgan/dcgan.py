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
        (X_train, _), (X_test, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = X_train / 255
        X_test = X_test / 255

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
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Keras-GAN/dcgan/images/cifar_%d.png" % epoch)
        plt.close()


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

    def featuremap_mia(self,
                x_in,
                x_out,
                plot_graph=False):
        """
        Take 50% of the training data and feed it through the neural network
        """
        x_in, x_out = np.reshape(x_in, (-1, 28, 28, 1)), np.reshape(x_out, (-1, 28, 28, 1))

        # gan_disc_model = self.get_gan_discriminator()
        # y_pred_in, y_pred_out = np.abs(gan_disc_model.predict(x_in)), np.abs(gan_disc_model.predict(x_out))

        self.featuremap_discriminator = self.get_featuremap_discriminator()
        y_pred_in, y_pred_out = self.featuremap_discriminator.predict(x_in), self.featuremap_discriminator.predict(x_out)

        def train_discriminator(y_pred_in, y_pred_out, validation_data):
            if self.featuremap_attacker is None:
                model = Sequential()
                model.name = "featuremap_mia"

                model.add(Dense(input_shape=(y_pred_in.shape[1:]), units=500))
                model.add(Dropout(0.2))
                model.add(Dense(units=250))
                model.add(Dropout(0.2))
                model.add(Dense(units=10))
                model.add(Dense(units=1, activation="sigmoid"))

                model.compile(optimizer="Adam",
                              metrics=["accuracy"],
                              loss="binary_crossentropy")
                self.featuremap_attacker = model
            self.featuremap_attacker.fit(np.concatenate((y_pred_in, y_pred_out), axis=0),
                np.concatenate((np.zeros(len(y_pred_in)), np.ones(len(y_pred_out)))),
                validation_data=validation_data,
                epochs=self.featuremap_mia_epochs,
                verbose=1)
            return self.featuremap_attacker

        n = int(len(y_pred_in)/2)
        val_data = (np.concatenate((y_pred_in[n:], y_pred_out[n:]), axis=0),
                    np.concatenate((np.zeros(len(y_pred_in[n:])), np.ones(len(y_pred_out[n:])))))
        c_disc = train_discriminator(y_pred_in[:n], y_pred_out[:n], val_data)
        y_pred_in = c_disc.predict(y_pred_in[n:])
        y_pred_out = c_disc.predict(y_pred_out[n:])

        # Get the accuracy for both approaches
        x = np.linspace(0,1, 100)
        y_acc = []  # Total accuracy per threshold
        y_sel = []  # Ratio of dataset that has a confidence score greater than threshold
        for thresh in x:
            accuracy_in = np.where(y_pred_in >= thresh)[0]  # Correctly captured
            accuracy_out = np.where(y_pred_out < thresh)[0]  # Correctly captured
            selected_samples = np.where((np.concatenate((y_pred_in, y_pred_out), axis=0) >= thresh))[0]

            total_acc = (len(accuracy_in) + len(accuracy_out)) / (len(y_pred_in) + len(y_pred_out))
            total_samples = len(selected_samples) / (len(y_pred_in) + len(y_pred_out))

            y_acc.append(total_acc)
            y_sel.append(total_samples)

        max_acc = max(y_acc)
        print("Featuremap Maximum Accuracy: {}".format(max_acc))

        if plot_graph:
            plt.title("[Featuremap] Membership Inference Accuracy")
            plt.xlabel("Threshold")
            plt.ylabel("Ratio")
            plt.plot(x, y_acc, label="Membership Inference Accuracy")
            plt.plot(x, y_sel, label="Positive samples")
            plt.legend()
            plt.show()

        return max_acc


    def distance_mia(self,
                x_in,
                x_out,
                plot_graph=False):
        """
        Membership inference based on output distance of images generated by the GAN and the two datasets
        """
        # Generate inputs from the generator
        n_generated_images = 100
        noise = np.random.normal(0, 1, (n_generated_images, self.latent_dim))
        sampled_labels = np.random.randint(0, 10, (n_generated_images, 1))
        gen_imgs = np.reshape(self.generator.predict([noise, sampled_labels]), (-1, 28, 28))

        def get_distances(x_data, gen_imgs):
            distances = []
            for suspect_img in x_data:
                min_distance = np.inf
                for gen_img in gen_imgs:
                    dist = np.linalg.norm(suspect_img-np.reshape(gen_img, (28, 28)), ord=2)
                    if dist < min_distance:
                        min_distance = dist
                distances.append(min_distance)
            return distances

        x_dist_in = get_distances(x_in, gen_imgs)
        x_dist_out = get_distances(x_out, gen_imgs)

        print("Mean Dist in: {}".format(np.mean(x_dist_in)))
        print("Mean Dist out: {}".format(np.mean(x_dist_out)))

        # Get the accuracy for both approaches
        x = np.linspace(*self.linspace_triplets_dist)
        y_acc = []  # Total accuracy per threshold
        y_sel = []  # Ratio of dataset that has a confidence score greater than threshold
        for thresh in x:
            accuracy_in = np.where(x_dist_in <= thresh)[0]  # Correctly captured
            accuracy_out = np.where(x_dist_out > thresh)[0]  # Correctly captured
            selected_samples = np.where((np.concatenate((x_dist_in, x_dist_out), axis=0) >= thresh))[0]

            total_acc = (len(accuracy_in) + len(accuracy_out)) / (len(x_dist_in) + len(x_dist_out))
            total_samples = len(selected_samples) / (len(x_dist_in) + len(x_dist_out))

            y_acc.append(total_acc)
            y_sel.append(total_samples)

        max_acc = max(y_acc)
        print("Distance Maximum Accuracy: {}".format(max_acc))

        if plot_graph:
            plt.title("[Distance] Membership Inference Accuracy")
            plt.xlabel("Threshold")
            plt.ylabel("Ratio")
            plt.plot(x, y_acc, label="Membership Inference Accuracy")
            plt.plot(x, y_sel, label="Positive samples")
            plt.legend()
            plt.show()

        return max_acc

    def logan_mia(self,
                  x_in,
                  x_out,
                  plot_graph=False):
        """
        Membership inference attack with the LOGAN paper
        @:param x_in The images in the dataset
        @:param x_out The images out of the dataset
        """
        x_in, x_out = np.reshape(x_in, (-1, 28, 28, 1)), np.reshape(x_out, (-1, 28, 28, 1))

        #gan_disc_model = self.get_gan_discriminator()
        #y_pred_in, y_pred_out = np.abs(gan_disc_model.predict(x_in)), np.abs(gan_disc_model.predict(x_out))

        logit_model = self.get_logit_discriminator()
        y_pred_in, y_pred_out = np.max(logit_model.predict(x_in), axis=1), np.max(logit_model.predict(x_out), axis=1)

        print(y_pred_in.mean())
        print(y_pred_out.mean())

        # Get the accuracy for both approaches
        x = np.linspace(*self.linspace_triplets_logan)
        y_acc = [] # Total accuracy per threshold
        y_sel = [] # Ratio of dataset that has a confidence score greater than threshold
        for thresh in x:
            accuracy_in = np.where(y_pred_in >= thresh)[0]   # Correctly captured
            accuracy_out = np.where(y_pred_out < thresh)[0]  # Correctly captured
            selected_samples = np.where((np.concatenate((y_pred_in, y_pred_out), axis=0) >= thresh))[0]

            total_acc = (len(accuracy_in)+len(accuracy_out))/(len(y_pred_in) + len(y_pred_out))
            total_samples = len(selected_samples)/(len(y_pred_in) + len(y_pred_out))

            y_acc.append(total_acc)
            y_sel.append(total_samples)

        max_acc = max(y_acc)
        print("LOGAN Maximum Accuracy: {}".format(max_acc))

        if plot_graph:
            plt.title("[LOGAN] Membership Inference Accuracy")
            plt.xlabel("Threshold")
            plt.ylabel("Ratio")
            plt.plot(x, y_acc, label="Membership Inference Accuracy")
            plt.plot(x, y_sel, label="Positive samples")
            plt.legend()
            plt.show()

        return max_acc

    def execute_featuremap_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000 + n]
        max_acc = self.featuremap_mia(x_in, x_out)

    def execute_dist_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000 + n]
        max_acc = self.distance_mia(x_in, x_out)

        with open('Keras-GAN/dcgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def execute_logan_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000+n]
        max_acc = self.logan_mia(x_in, x_out)

        with open('Keras-GAN/dcgan/logs/logan_mia.csv', mode='a') as file_:
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

        load(self.generator, "generator")
        load(self.discriminator, "discriminator")

    def save_model(self):

        def save(model, model_name):
            model_path = "Keras-GAN/dcgan/saved_model/%s.json" % model_name
            weights_path = "Keras-GAN/dcgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

if __name__ == '__main__':
    (X_train, y_train) = extract_training_samples('digits')

    dcgan = DCGAN()
    dcgan.load_model()

    n = 500
    x_in, x_out = X_train[0:n], X_train[100000:100000+n]
    x_in = x_in.astype(np.float32)-127.5/127.5
    x_out = x_out.astype(np.float32) - 127.5 / 127.5

    dcgan.featuremap_mia(x_in, x_out, plot_graph=True)
    dcgan.distance_mia(x_in, x_out, plot_graph=True)
    dcgan.logan_mia(x_in, x_out, plot_graph=True)