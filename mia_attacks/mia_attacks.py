import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt


def featuremap_mia(featuremap_discriminator,
                   featuremap_attacker,
                   epochs,
                   x_in,
                   x_out,
                   val_in,
                   val_out,
                   plot_graph=False):
    """
    Take 50% of the training data and feed it through the neural network
    @:param featuremap_discriminator Returns features for an input from the source model
    @;param featuremap_attacker: The attacker model (can be None in the first run)
    @:param epochs The number of epochs to train the attack model
    """

    y_pred_in, y_pred_out = featuremap_discriminator.predict(x_in), featuremap_discriminator.predict(x_out)
    y_val_in, y_val_out = featuremap_discriminator.predict(x_in), featuremap_discriminator.predict(x_out)

    def train_discriminator(y_pred_in,
                            y_pred_out,
                            validation_data,
                            featuremap_attacker):
        if featuremap_attacker is None:
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
            featuremap_attacker = model
        featuremap_attacker.fit(np.concatenate((y_pred_in, y_pred_out), axis=0),
                                np.concatenate((np.zeros(len(y_pred_in)), np.ones(len(y_pred_out)))),
                                validation_data=validation_data,
                                epochs=epochs,
                                verbose=1)
        return featuremap_attacker

    val_data = (np.concatenate((y_val_in, y_val_out), axis=0),
                np.concatenate((np.zeros(len(y_val_in)), np.ones(len(y_val_out)))))
    c_disc = train_discriminator(y_pred_in, y_pred_out, val_data, featuremap_attacker)

    y_pred_in = c_disc.predict(featuremap_discriminator.predict(val_in))
    y_pred_out = c_disc.predict(featuremap_discriminator.predict(val_out))

    # Get the accuracy for both approaches
    x = np.linspace(y_pred_in.min(), y_pred_in.max(), 500)
    y_acc = []  # Total accuracy per threshold
    y_sel = []  # Ratio of dataset that has a confidence score greater than threshold
    for thresh in x:
        accuracy_in = np.where(y_pred_in <= thresh)[0]  # Correctly captured
        accuracy_out = np.where(y_pred_out >= thresh)[0]  # Correctly captured
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

    return c_disc


def distance_mia(generator,
                 x_in,
                 x_out,
                 plot_graph=False,
                 targeted=False,
                 verbose=True):
    """
    Membership inference based on output distance of images generated by the GAN and the two datasets
    """
    img_shape = x_in.shape[1:]

    n_generated_images = 100
    latent_dim = 100

    # Generate inputs from the generator
    noise = np.random.normal(0, 1, (n_generated_images, latent_dim))
    if targeted:
        sampled_labels = np.random.randint(0, 10, (n_generated_images, 1))
        gen_imgs = generator.predict([noise, sampled_labels]).squeeze()
    else:
        gen_imgs = generator.predict([noise]).squeeze()

    def get_distances(x_data, gen_imgs):
        distances = []
        for suspect_img in x_data:
            min_distance = np.inf
            for gen_img in gen_imgs:
                dist = 0
                for i in range(gen_img.shape[-1]):
                    dist += np.linalg.norm(suspect_img[:,:,i] - gen_img[:,:,i], ord=2)
                if dist < min_distance:
                    min_distance = dist
            distances.append(min_distance)
        return distances

    x_dist_in = get_distances(x_in, gen_imgs)
    x_dist_out = get_distances(x_out, gen_imgs)

    if verbose:
        print("Mean Dist in: {}".format(np.mean(x_dist_in)))
        print("Mean Dist out: {}".format(np.mean(x_dist_out)))

    # Get the accuracy for both approaches
    x = np.linspace(min(x_dist_in), max(x_dist_out), 1000)
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


def logan_mia(logit_model,
              x_in,
              x_out,
              plot_graph=False,
              verbose=True):
    """
    Membership inference attack with the LOGAN paper
    @:param Model with logit output
    @:param x_in The images in the dataset
    @:param x_out The images out of the dataset
    """
    # Use class prediction output (not used in all GANs)
    # gan_disc_model = self.get_gan_discriminator()
    # y_pred_in, y_pred_out = np.abs(gan_disc_model.predict(x_in)), np.abs(gan_disc_model.predict(x_out))

    y_pred_in, y_pred_out = logit_model.predict(x_in), logit_model.predict(x_out)

    if verbose:
        print("[LOGAN] X_in mean: {} X_out mean: {}".format(np.mean(y_pred_in), np.mean(y_pred_out)))

    # Get the accuracy for both approaches
    x = np.linspace(min(y_pred_in + y_pred_out), max(y_pred_in+y_pred_out), 1000)

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
    print("[LOGAN] Maximum Accuracy: {}".format(max_acc))

    if plot_graph:
        plt.title("[LOGAN] Membership Inference Accuracy")
        plt.xlabel("Threshold")
        plt.ylabel("Ratio")
        plt.plot(x, y_acc, label="Membership Inference Accuracy")
        plt.plot(x, y_sel, label="Positive samples")
        plt.legend()
        plt.show()

    return max_acc
