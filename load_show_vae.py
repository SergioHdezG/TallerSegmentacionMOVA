#!/usr/bin/env python
# coding: utf-8
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras import backend as K
from tensorflow import keras
from keras.engine import data_adapter

import os
import numpy as np
import matplotlib.pyplot as plt

# Esta función reordena la clases de la máscara para visualizar mejor los datos

def reorder_visulization(img_label):
    mask_aux = np.copy(img_label[:, :, 0])
    img_label[:, :, 0] = img_label[:, :, 1]
    img_label[:, :, 1] = img_label[:, :, 3]
    mask_aux2 = np.copy(img_label[:, :, 2])
    img_label[:, :, 2] = img_label[:, :, 4]
    img_label[:, :, 3] = img_label[:, :, 5]
    img_label[:, :, 4] = img_label[:, :, 6]
    img_label[:, :, 5] = mask_aux2
    img_label[:, :, 6] = mask_aux
    return img_label

#
data_path = "cityscapes"

x = os.listdir(os.path.join(data_path, "train_images_npy"))
y = os.listdir(os.path.join(data_path, "train_masks_npy"))

# Ordenamos los nombres delos ficheros por sequencia
x.sort(key=lambda x: (x.split("f")[-1].split("_")[0], int(x.split("_")[1])))
y.sort(key=lambda x: (x.split("f")[-1].split("_")[0], int(x.split("_")[1])))

x = np.array(x)
y = np.array(y)

print('Training size: ', len(x))
print(x[:5])
print(y[:5])


# Cargamos los datos de entrenamiento en un numpy array.

x = x[:100]
y = y[:100]
x_train = [np.load(os.path.join(data_path, "train_images_npy", name)) for name in x]
y_train = [np.load(os.path.join(data_path, "train_masks_npy", name)) for name in y]
print('Training size: ', len(x))

# Se convierte x_train a un numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# Esta función reordena la clases de la máscara para visualizar mejor los datos

n_img = np.random.randint(0, x_train.shape[0])
img_in = x_train[n_img]
img_label = y_train[n_img]

print("Train images range: : [", np.amin(img_in), ", ", np.amax(img_in), "]")
print("Train label images range: : [", np.amin(img_label[:, :, 0]), ",", np.amax(img_label[:, :, 0]), "]")

print('Input image shape: ', img_in.shape)
print('Label image shape: ', img_label.shape)

print("Train images range: : [", np.amin(img_in), ", ", np.amax(img_in), "]")

img_in = x_train[0]
img_label = y_train[0]
print('Input image shape: ', img_in.shape)
print('Label image shape: ', img_label.shape)

print("Train images range: : [", np.amin(img_in), ", ", np.amax(img_in), "]")
print("Train label images range: : [", np.amin(img_label), ",", np.amax(img_label), "]")

# separamos un subconjunto para test
test_split = 0.15
test_split = int(x_train.shape[0] * test_split)
test_idx = np.random.choice(x_train.shape[0], test_split, replace=False)  # Seleccionamos varios índices aleatorios
train_mask = np.array([False if i in test_idx else True for i in
                       range(x_train.shape[0])])  # Creamos una máscara para seleccionar los datos de entrenamiento

x_test = x_train[test_idx]  # Seleccionamos datos de test
y_test = y_train[test_idx]

x_train = x_train[train_mask]  # Seleccionamos datos de entrenamiento
y_train = y_train[train_mask]

print('Train samples: ', x_train.shape[0])
print('Test samples: ', x_test.shape[0])

# separamos un subconjunto de validación

val_split = 0.05
val_split = int(x_train.shape[0] * val_split)
val_idx = np.random.choice(x_train.shape[0], val_split, replace=False)  # Seleccionamos varios índices aleatorios
train_mask = np.array([False if i in val_idx else True for i in range(x_train.shape[0])])  # Creamos una máscara para seleccionar los datos de entrenamiento

x_val = x_train[val_idx]  # Seleccionamos datos de validacón
y_val = y_train[val_idx]

x_train = x_train[train_mask]  # Seleccionamos datos de entrenamiento
y_train = y_train[train_mask]

print('Train samples: ', x_train.shape[0])
print('Validation samples: ', x_val.shape[0])

def encoder(latent_dim, input_shape):
    inputs = Input(shape=input_shape, name='encoder_input')

    # Construimos el modelo de codificador
    x = Conv2D(filters=128,
               kernel_size=3,
               activation='relu',
               strides=2,
               padding='same')(inputs)

    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=128,
               kernel_size=3,
               activation='relu',
               strides=2,
               padding='same')(x)

    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=256,
               kernel_size=3,
               activation='relu',
               strides=2,
               padding='same')(x)

    # x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=256,
               kernel_size=3,
               activation='relu',
               strides=2,
               padding='same')(x)

    x = Conv2D(filters=256,
               kernel_size=3,
               activation='relu',
               strides=2,
               padding='same')(x)

    # Necesitamos la información de las dimensiones de la salida de la última
    # capa convolucional para poder reconstruir la imagen al tamaño correcto en
    # el decodificador
    last_conv_shape = K.int_shape(x)

    # Generamos la huella latente
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x)
    z_log_var = Dense(latent_dim, name='z_log_var', activation='linear')(x)

    # Par usar una función externa en un modelo de Keras tenemos que hacerlo a 
    # través de una capa Lambda. La función sampling realiza el truco de la reparametrización
    # z = Lambda(sampling, name='z')([z_mean, z_log_var])
    z = Sampling()([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    return encoder, last_conv_shape


# In[16]:


def decoder(latent_dim, last_conv_shape, out_channels):
    # Construimos el modelo de decodificador
    latent_input = Input(shape=(latent_dim,), name='z_sampled')

    # shape = 4*8*8 = 584
    x = Dense(last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3], activation='relu')(latent_input)
    # shape = 584 => (4, 8, 8)
    x = Reshape((last_conv_shape[1], last_conv_shape[2], last_conv_shape[3]))(x)

    x = Conv2DTranspose(filters=256,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    x = Conv2DTranspose(filters=256,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    # x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(filters=256,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    # x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    # x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same')(x)

    output = Conv2DTranspose(filters=out_channels,
                          kernel_size=3,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)  # Activación sigmoide para que las salidas estén en rango [0, 1]



    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_input, output, name='decoder')
    decoder.summary()

    return decoder


# Vamos a contruir un modelo de Keras con el Encoder y el Decoder.

# In[17]:


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # Factor de importancia de la regulación (beta-VAE)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

        self.reconstruction_loss = None

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction))
            )

        total_loss, kl_loss = vae_loss_func(reconstruction_loss, z_log_var, z_mean, beta=beta)

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(beta*kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "beta*kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)


        # Compute predictions
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        # Updates the metrics tracking the loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(y, reconstruction))
        )

        total_loss, kl_loss = vae_loss_func(reconstruction_loss, z_log_var, z_mean, beta=beta)

        # Update the metrics.
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(beta*kl_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "beta*kl_loss": self.val_kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def vae_loss_func(reconstruction_loss, z_log_var, z_mean, beta=0.5):
    # Vamos a calcular la Kullback-Leibler dirvegence entre dos gaussianas
    # KL = 1/2 {(std_0/std_1)^2 + (mu_1-mu_0)^2/std_1^2 - 1 + ln(st_1^2/st_0^2)}
    # Se simplifica teniendo en cuenta que:
    # mu_0 = 0; std_0 = 1; std_1 = e^{log(std_1^2)/2} = e^z_log_var
    kl_loss = - 0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    # beta: factor de importancia de la regulación (beta-VAE)
    vae_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

    return vae_loss, kl_loss

input_shape = (256, 512, 3)
out_channels= 7
latent_dim = 2048
beta = 1000

encoder, last_conv_shape = encoder(latent_dim, input_shape)
decoder = decoder(latent_dim, last_conv_shape, out_channels)

# vae.compile(optimizer=keras.optimizers.Adam(lr=1e-4))

batch_size = 64
# epochs = 50

# vae.fit(x_train, y_train,
#        epochs=epochs,
#        batch_size=batch_size,
#        shuffle=True,
#        validation_data=(x_val, y_val))
encoder_path = '/media/archivos/home/PycharmProjects2022/TallerSegmentacion/checkpoints/taller_encoder/weights.h5'
decoder_path = '/media/archivos/home/PycharmProjects2022/TallerSegmentacion/checkpoints/taller_decoder/weights.h5'
vae_path = '/media/archivos/home/PycharmProjects2022/TallerSegmentacion/checkpoints/taller_vae/weights.h5'
# # Si no se hubiese creado el modelo todavía habria que crear uno nuevo con la siguiente linea
# # encoder, decoder, vae = build_model(input_shape, out_channels, latent_dim)

# # load weights into new model
encoder.load_weights(encoder_path)

# # load weights into new model
decoder.load_weights(decoder_path)

# # Si fuesemos a reentrenar necesitariamos compilamos el modelo de nuevo
vae = VAE(encoder, decoder, beta)
vae.compile()

# # Testing
vae.evaluate(x_test, y_test, batch_size=batch_size)

figsize = 5
num_examples_to_generate = figsize*figsize
fig = plt.figure(figsize=(20, 10))
fig.suptitle('Resultados segmentación')
examples_index = np.random.choice(x_test.shape[0], figsize*2)
examples = x_test[examples_index]
z_mean, z_log_var, z = encoder.predict(examples)
predictions = decoder.predict(z)


for i in range(figsize*2):
    img_label = reorder_visulization(predictions[i])
    preds = np.argmax(img_label, axis=-1) / 6.
    plt.subplot(4, figsize, i+1)
    plt.imshow(examples[i])
    plt.subplot(4, figsize, i + 1 + figsize*2)
    plt.imshow(preds)
plt.colorbar()
plt.axis('off')
plt.show()

rnd = random.randint(0, predictions.shape[0]-1)
img_label = reorder_visulization(predictions[rnd])

plt.figure(figsize=(15, 15))
ax = plt.subplot(4, 3, 1)
ax.set_title("imagen")
plt.imshow(examples[rnd])


# Mostramos cada máscara por separado
ax = plt.subplot(4, 3, 2)
ax.set_title("vehículo")
plt.imshow(img_label[:, :, 0])

ax = plt.subplot(4, 3, 3)
ax.set_title("persona")
plt.imshow(img_label[:, :, 1])

ax = plt.subplot(4, 3, 4)
ax.set_title("edificios")
plt.imshow(img_label[:, :, 2])

ax = plt.subplot(4, 3, 5)
ax.set_title("vegetación")
plt.imshow(img_label[:, :, 3])

ax = plt.subplot(4, 3, 6)
ax.set_title("cielo")
plt.imshow(img_label[:, :, 4])

ax = plt.subplot(4, 3, 7)
ax.set_title("suelo")
plt.imshow(img_label[:, :, 5])

ax = plt.subplot(4, 3, 8)
ax.set_title("fondo")
plt.imshow(img_label[:, :, 6])

# Generamos una imágen con todas las máscaras
img_all_label = np.argmax(img_label, axis=-1) / 6.

ax = plt.subplot(4, 3, 9)
ax.set_title("mapa segmentación")
plt.imshow(img_all_label)
plt.colorbar()

plt.show()

random_vector_for_generation = np.random.normal(size=(num_examples_to_generate, latent_dim))

predictions = decoder.predict(random_vector_for_generation)

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Nuevos ejemplos generados')

for i in range(num_examples_to_generate):
    img_label = reorder_visulization(predictions[i])
    preds = np.argmax(img_label, axis=-1) / 6.
    plt.subplot(figsize, figsize, i+1)
    plt.imshow(preds)
plt.colorbar()
plt.axis('off')
plt.show()





