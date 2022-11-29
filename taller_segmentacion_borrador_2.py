from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib import image as plt_image


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
from tensorflow.python.keras.models import model_from_json


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def generate_and_save_images(models, latent_dim, test_data):
    figsize = 5
    num_examples_to_generate = figsize*figsize
    encoder, decoder = models

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = np.random.normal(size=(num_examples_to_generate, latent_dim))

    predictions = decoder.predict(random_vector_for_generation)

    predictions = np.argmax(predictions, axis=-1)
    fig1 = plt.figure(figsize=(figsize, figsize))
    fig1.suptitle('Nuevos ejemplos generados')
    for i in range(num_examples_to_generate):
        plt.subplot(figsize, figsize, i+1)
        plt.imshow(predictions[i])

        # if predictions.shape[3] == 1:
        #     plt.imshow(predictions[i, :, :], cmap='gray')
        # else:
        #     plt.imshow(predictions[i, :, :, :])
    plt.axis('off')
    # plt.savefig('figura_6.png')

    fig2 = plt.figure()
    fig2.suptitle('Resultados segmentación')
    examples_index = np.random.choice(test_data.shape[0], figsize*2)
    examples = test_data[examples_index]
    z_mean, z_log_var, z = encoder.predict(examples)
    predictions = decoder.predict(z_mean)

    predictions = np.argmax(predictions, axis=-1)

    for i in range(figsize*2):

        plt.subplot(4, figsize, i+1)
        plt.imshow(examples[i])
        # if examples.shape[3] == 1:
        #     plt.imshow(examples[i, :, :, 0], cmap='gray')
        # else:
        #     plt.imshow(examples[i])

        plt.subplot(4, figsize, i + 1 + figsize*2)
        plt.imshow(predictions[i], cmap='gray')

        # if examples.shape[3] == 1:
        #     plt.imshow(predictions[i, :, :, 0], cmap='gray')
        # else:
        #     plt.imshow(predictions[i])


    plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.savefig('figura_7.png')
    plt.show()



# MNIST dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

kernel_size = 3
latent_dim = 128

def build_model(input_shape=(128, 128, 3), out_channels=1):
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=64,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

    x = Conv2D(filters=64,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

    x = Conv2D(filters=128,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

    x = Conv2D(filters=256,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)


    x = Conv2DTranspose(filters=256,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    x = Conv2DTranspose(filters=128,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    x = Conv2DTranspose(filters=64,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    x = Conv2DTranspose(filters=64,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    outputs = Conv2DTranspose(filters=out_channels,
                          kernel_size=kernel_size,
                          activation='linear',
                          padding='same',
                          name='decoder_output')(x)

    outputs = tf.keras.activations.softmax(outputs, axis=-1)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])

    labels = Input(shape=input_shape, name='lables')

    vae = Model([inputs, labels], outputs, name='vae')
    # reconstruction_loss = binary_crossentropy(K.flatten(inputs),
    #                                           K.flatten(outputs))
    #
    # reconstruction_loss *= input_shape[0] * input_shape[1]
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # beta = 10
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + beta*kl_loss)

    vae_loss = reconstruction_loss(labels, outputs, z_log_var, z_mean, input_shape)
    vae.add_loss(vae_loss)


    vae.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))
    vae.summary()

    return encoder, decoder, vae

def reconstruction_loss(y_true, y_pred, z_log_var, z_mean, input_shape):
    reconstruction_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    # reconstruction_loss /= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = - K.mean(kl_loss, axis=-1)
    beta = 0.5
    # kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + beta * kl_loss)

    return vae_loss

def load_model(encoder_path, decoder_path):
    encoder, decoder, vae = build_model()

    # json_file = open(name+'.json', 'r')
    # loaded_model_json = json_file.read()
    # encoder = model_from_json(loaded_model_json)
    # json_file.close()

    # load weights into new model
    encoder.load_weights(encoder_path + ".h5")
    # encoder.compile(loss='mse', optimizer=Adam(lr=s

    # json_file = open(name+'.json', 'r')
    # loaded_model_json = json_file.read()
    # decoder = model_from_json(loaded_model_json)
    # json_file.close()

    # load weights into new model
    decoder.load_weights(decoder_path + ".h5")

    return encoder, decoder, vae

def test(x_test):
    encoder, decoder, vae = load_model('/environments/carla/encoder_64_64_128_256_l128',
                                       '/environments/carla/decoder_64_64_128_256_l128')
    generate_and_save_images((encoder, decoder), latent_dim, x_test)

def train(x_train, y_train, x_val, y_val, epochs, batch_size):
    encoder, decoder, vae = build_model(input_shape=(128, 128, 3), out_channels=2)

    models = (encoder, decoder)

    # train the autoencoder
    vae.fit([x_train, y_train], y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=([x_val, y_val], y_val))

    data_path = "/home/serch/TFM/CAPOIRL-TF2/dataset_seg/ConferenceVideoSegmentationDataset"
    testing_in = os.listdir(os.path.join(data_path, "original_testing"))
    testing_label = os.listdir(os.path.join(data_path, "ground_truth_testing"))
    # Ordenamos las imágenes por sequencia
    testing_in.sort(key=lambda x: (int(x.split("_")[0].split("l")[1]), int(x.split(".")[0].split("_")[1])))
    testing_label.sort(key=lambda x: (int(x.split("_")[0].split("t")[1]), int(x.split(".")[0].split("_")[1])))
    x_test = [plt_image.imread(os.path.join(data_path, "original_testing", name)) for name in testing_in[:]]
    y_test = [plt_image.imread(os.path.join(data_path, "ground_truth_testing", name)) for name in testing_label[:]]

    for i in range(len(x_test)):
        x_test[i] = cv2.resize(x_test[i], (128, 128))
        y_test[i] = cv2.resize(y_test[i], (128, 128))

    # Normalizamos los datos de entrada en rango [0, 1]
    x_test = np.array(x_test) / 255.
    y_test = np.array(y_test) / 255.
    y_test = np.expand_dims(y_test[:, :, :, 1], axis=-1)

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)  # one hot encoding

    print('Testing step')
    vae.evaluate([x_test, y_test], y_test, batch_size=batch_size)
    # # serialize model to JSON
    # save_name = '/home/serch/TFM/IRL2/CAPORL/environments/carla/carla_encoder'
    # model_json = encoder.to_json()
    # with open(save_name + ".json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # encoder.save_weights(save_name + ".h5")
    # print("Encoder saved  to disk")
    #
    # save_name = '/home/serch/TFM/IRL2/CAPORL/environments/carla/carla_decoder'
    # model_json = decoder.to_json()
    # with open(save_name + ".json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # decoder.save_weights(save_name + ".h5")
    # print("Decoder saved  to disk")

    generate_and_save_images(models, latent_dim, x_test)

def main():
    print("Loading dataset..")
    images = np.load('/home/serch/TFM/IRL3/environments/carla/dataset/rgb_carla.npy')

    data_path = "/home/serch/TFM/CAPOIRL-TF2/dataset_seg/ConferenceVideoSegmentationDataset"
    training_in = os.listdir(os.path.join(data_path, "original_training"))
    training_label = os.listdir(os.path.join(data_path, "ground_truth_training"))

    # Ordenamos las imágenes por sequencia
    training_in.sort(key=lambda x: (int(x.split("_")[0].split("l")[1]), int(x.split(".")[0].split("_")[1])))
    training_label.sort(key=lambda x: (int(x.split("_")[0].split("t")[1]), int(x.split(".")[0].split("_")[1])))

    print('Training size: ', len(training_in))
    print(training_in[:10])
    print(training_label[:10])

    # Cargamos todas las imágenes a un numpy array

    x_train = [plt_image.imread(os.path.join(data_path, "original_training", name)) for name in training_in[:]]
    y_train = [plt_image.imread(os.path.join(data_path, "ground_truth_training", name)) for name in training_label[:]]


    for i in range(len(x_train)):
        x_train[i] = cv2.resize(x_train[i], (128, 128))
        y_train[i] = cv2.resize(y_train[i], (128, 128))

    # Normalizamos los datos de entrada en rango [0, 1]
    x_train = np.array(x_train) / 255.
    y_train = np.array(y_train) / 255.
    y_train = np.expand_dims(y_train[:, :, :, 1], axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)  # one hot encoding

    val_split = 0.15
    val_split = int(x_train.shape[0] * val_split)
    val_idx = np.random.choice(x_train.shape[0], val_split, replace=False)
    train_mask = np.array([False if i in val_idx else True for i in range(x_train.shape[0])])

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    # network parameters
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    batch_size = 128
    epochs = 50

    import gc
    gc.collect()
    train(x_train, y_train, x_val, y_val, epochs, batch_size)


main()