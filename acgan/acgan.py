from __future__ import print_function, division

from emnist import extract_training_samples
from keras import activations
from keras.engine.saving import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class ACGAN():
    def __init__(self,
                 n_samples=50000,
                 linspace_triplets_logan=(0, 200, 300),
                 log_logan_mia=False,
                 featuremap_mia_epochs=100,
                 log_featuremap_mia=False,
                 linspace_triplets_dist=(1800, 2300, 1000),
                 log_dist_mia=False,
                 load_model=False):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100
        self.log_logan_mia = log_logan_mia
        self.log_dist_mia = log_dist_mia
        self.featuremap_mia_epochs=featuremap_mia_epochs
        self.log_featuremap_mia=log_featuremap_mia
        self.n_samples = n_samples
        self.linspace_triplets_logan = linspace_triplets_logan
        self.linspace_triplets_dist = linspace_triplets_dist

        self.X_train, self.y_train = extract_training_samples('digits')
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        if load_model:
            self.load_model()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

        self.logit_discriminator = None
        self.gan_discriminator = None
        self.featuremap_discriminator = None
        self.featuremap_attacker = None

    def build_generator(self):

        model = Sequential()
        model.name = "generator"

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.name = "discriminator"

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid", name="logits")(features)
        label = Dense(self.num_classes, activation="softmax", name="softmax")(features)

        return Model(img, [validity, label])

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

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        if self.X_train is None:
            self.X_train, self.y_train = extract_training_samples('digits')
        X_train, y_train = self.X_train[:self.n_samples], self.y_train[:self.n_samples]

        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9 
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)
                if self.log_logan_mia:
                    self.execute_logan_mia()
                if self.log_dist_mia:
                    self.execute_dist_mia()
                if epoch > 3000 and self.log_featuremap_mia:
                    self.execute_featuremap_mia()

    def execute_featuremap_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000 + n]
        max_acc = self.featuremap_mia(x_in, x_out)

    def execute_dist_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000 + n]
        max_acc = self.distance_mia(x_in, x_out)

        with open('Keras-GAN/acgan/logs/dist_mia.csv', mode='a') as file_:
            file_.write("{}".format(max_acc))
            file_.write("\n")

    def execute_logan_mia(self):
        n = min(self.n_samples, 1000)
        x_in, x_out = self.X_train[0:n], self.X_train[100000:100000+n]
        max_acc = self.logan_mia(x_in, x_out)

        with open('Keras-GAN/acgan/logs/logan_mia.csv', mode='a') as file_:
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
        fig.savefig("Keras-GAN/acgan/images/%d.png" % epoch)
        plt.close()
        return gen_imgs

    def load_model(self):

        def load(model, model_name):
            weights_path = "Keras-GAN/acgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_weight": weights_path}
            model.load_weights(options['file_weight'])

        load(self.generator, "generator")
        load(self.discriminator, "discriminator")

    def save_model(self):

        def save(model, model_name):
            model_path = "Keras-GAN/acgan/saved_model/%s.json" % model_name
            weights_path = "Keras-GAN/acgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    (X_train, y_train) = extract_training_samples('digits')

    acgan = ACGAN()
    acgan.load_model()

    n = 500
    x_in, x_out = X_train[0:n], X_train[100000:100000+n]
    x_in = x_in.astype(np.float32)-127.5/127.5
    x_out = x_out.astype(np.float32) - 127.5 / 127.5

    acgan.featuremap_mia(x_in, x_out, plot_graph=True)
    acgan.distance_mia(x_in, x_out, plot_graph=True)
    acgan.logan_mia(x_in, x_out, plot_graph=True)
