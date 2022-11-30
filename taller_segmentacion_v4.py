# -*- coding: utf-8 -*-
"""taller segmentacion VAE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WSnMKJSnbc2xUiZxOQ13r2GDRHvu4P1M
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import os
import cv2

import numpy as np
from matplotlib import image as plt_image
import matplotlib.pyplot as plt
import argparse
import re

from tensorflow.python.keras.models import model_from_json

"""# Preprocesado de datos

En primer lugar tenemos que obtener acceso a nuestro google drive

"""

# from google.colab import drive
# drive.mount('/content/drive')
#
# """Buscamos la url del datset dentro de nuestro drive y lo descomprimimos el dataset en la url indicadas"""
#
# !unzip -u "/content/drive/MyDrive/Colab Notebooks/teller segmentacion/ConferenceVideoSegmentationDataset.zip" -d "/content/drive/MyDrive/Colab Notebooks/teller segmentacion/tmp"


data_path = "/home/serch/TFM/CAPOIRL-TF2/dataset_seg/oxfordpets/"

training_input = []
training_label = []
test_input = []
test_label = []

x = os.listdir(os.path.join(data_path, "images"))
y = os.listdir(os.path.join(data_path, "annotations/trimaps"))

x_aux = []
y_aux = []
for sample in x:
    if '.jpg' in sample:
        x_aux.append(sample)

for sample in y:
    if '.png' in sample:
        y_aux.append(sample)

x = x_aux
y = y_aux

# Ordenamos las imágenes por sequencia
x.sort(key=lambda x: (x.split("_")[0], int(x.split(".")[0].split("_")[-1])))
y.sort(key=lambda x: (x.split("_")[0], int(x.split(".")[0].split("_")[-1])))

print('Training size: ', len(x))
print(x[:5])
print(y[:5])

"""Cargamos los datos de entrenamiento en un numpy array.

En este ejemplo ce cargan solo las 100 primeras imágenes.
"""

x_train = [plt_image.imread(os.path.join(data_path, "images", name)) for name in x[:800]]
y_train = [plt_image.imread(os.path.join(data_path, "annotations/trimaps", name)) for name in y[:800]]

print('Training size: ', len(x_train))

"""Para este caso de ejemplo vamos a ajustar el tamaño de los datos. Lo vamos a reducir desde su tamaño inicial (360, 640) a (128, 128) por tres motivos:


1.   Para ajustarlos al tamaño de entrada que nos interesa en la red neuronal
2.   Para que al realizar convoluciones con stride=2 no quede ninguna dimensión imapar. Si se diese ese caso luego tendríamos problemas para recuperar el tamaño original a la salida del autoencoder.
3.   Para simplificar y acelerar el procesamiento a efectos prácticos de este taller.



"""

for i in range(len(x_train)):
    x_train[i] = cv2.resize(x_train[i], (128, 128))
    y_train[i] = cv2.resize(y_train[i], (128, 128))

# Se convierte x_train a un numpy array y eliminamos el canal alpha que aparece en algunas imágenes.
x_train = np.array([x[:, :, :3] for x in x_train])

"""Realizamos una visualización de los datos para ver que tras la transformación anterior siguen siendo correctos."""

n_img = np.random.randint(0, x_train.shape[0])
img_in = x_train[n_img]
img_label = y_train[n_img]

print('Input image shape: ', img_in.shape)
print('Label image shape: ', img_label.shape)

plt.subplot(2, 2, 1)
plt.imshow(img_in)

plt.subplot(2, 2, 2)
plt.imshow(img_label)

n_img = np.random.randint(0, x_train.shape[0])
img_in = x_train[n_img]
img_label = y_train[n_img]

plt.subplot(2, 2, 3)
plt.imshow(img_in)

plt.subplot(2, 2, 4)
plt.imshow(img_label)

plt.show()

print("Train images range: : [", np.amin(img_in), ", ", np.amax(img_in), "]")
print("Test images range: : [", np.amin(img_label), ",", np.amax(img_label), "]")

"""Como se puede observar en la celda anterior, los datos se encuentran en el rango de valores [0, 255]. Nos interesa normalizarlos al rango [0, 1] y convertirlos a tipo float32.

Posteriormente nos quedamos solo con uno de los tres canales de las imágenes segmentadas ya que se encuentran triplicados.
"""

# Normalizamos los datos de entrada en rango [0, 1]
x_train = x_train / 255.
print(x_train.shape)

# transformamos los valores de y_train para que tengan información categórica en lugar de valores de intensidad (0.00392, 0.00784, 0.01176) -> (animal, fondo, borde)
y_train = np.array(y_train)
bins = np.array([0.0, 0.005, 0.01, 1.])
y_train = np.digitize(y_train, bins) - 1

# Expandimos las dimensiones de las eqtiquetas para que queden (w, h, 1)
y_train = np.expand_dims(y_train, axis=-1)
print(y_train.shape)

"""imagenes segmentadas a codificación one hot para que sean tratadas como etiquetas. De este modo la clase 0 se correspondera con el fondo de la imagen y la clase 1 con el primer plano."""
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)  # one hot encoding

"""Revisamos la forma y los valores de los datos. Las etiquetas ahora tienen que tener forma (h, w, 2)"""

img_in = x_train[0]
img_label = y_train[0]
print('Input image shape: ', img_in.shape)
print('Label image shape: ', img_label.shape)

print("Train images range: : [", np.amin(img_in), ", ", np.amax(img_in), "]")
print("Test images range: : [", np.amin(img_label), ",", np.amax(img_label), "]")

"""Separamos un subconjunto de los datos para hacer validación."""

# saparamos un subconjunto para test
test_split = 0.15
test_split = int(x_train.shape[0] * test_split)
test_idx = np.random.choice(x_train.shape[0], test_split, replace=False)  # Seleccionamos varios índices aleatorios
train_mask = np.array([False if i in test_idx else True for i in
                       range(x_train.shape[0])])  # Creamos una máscara para seleccionar los datos de entrenamiento

x_test = x_train[test_idx]  # Seleccionamos datos de test
y_test = y_train[test_idx]

x_train = x_train[train_mask]  # Seleccionamos datos de entrenamiento
y_train = y_train[train_mask]

# saparamos un subconjunto de validación
val_split = 0.15
val_split = int(x_train.shape[0] * val_split)
val_idx = np.random.choice(x_train.shape[0], val_split, replace=False)  # Seleccionamos varios índices aleatorios
train_mask = np.array([False if i in val_idx else True for i in
                       range(x_train.shape[0])])  # Creamos una máscara para seleccionar los datos de entrenamiento

x_val = x_train[val_idx]  # Seleccionamos datos de validacón
y_val = y_train[val_idx]

x_train = x_train[train_mask]  # Seleccionamos datos de entrenamiento
y_train = y_train[train_mask]

print('Train samples: ', x_train.shape[0])
print('Validation samples: ', x_val.shape[0])


def build_encoder(inputs, input_shape, kernel_size):
    # build encoder model
    x = inputs
    x = tf.keras.layers.LayerNormalization()(x)
    x = Conv2D(filters=64,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = Conv2D(filters=64,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = Conv2D(filters=128,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = Conv2D(filters=256,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

    # shape info needed to build decoder model
    last_conv_shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = Dense(last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3], activation='relu')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    return encoder, z_mean, z_log_var, last_conv_shape


def build_decoder(latent_dim, shape):
    # build decoder model
    latent_input = Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.LayerNormalization()(latent_input)
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = Conv2DTranspose(filters=256,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = Conv2DTranspose(filters=128,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = Conv2DTranspose(filters=64,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = Conv2DTranspose(filters=64,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)

    output = Conv2DTranspose(filters=out_channels,
                             kernel_size=kernel_size,
                             activation='linear',
                             padding='same',
                             name='decoder_output')(x)  # Activación sigmoide para que las salidas estén en rango [0, 1]

    output = tf.keras.activations.softmax(output, axis=-1)

    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_input, output, name='decoder')
    decoder.summary()

    return decoder


def build_model(input_shape=(128, 128, 3), out_channels=1, latent_dim=128, kernel_size=3):
    # AE model = encoder + decoder
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder, z_mean, z_log_var, last_conv_shape = build_encoder(inputs, input_shape, kernel_size)
    decoder = build_decoder(latent_dim, last_conv_shape)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])  # Pasamos "z" (encoder(inputs)[2]) el decodificador

    labels = Input(shape=(*input_shape[:-1], out_channels), name='lables')

    vae = tf.keras.models.Model([inputs, labels], outputs, name='vae')

    vae_loss = reconstruction_loss(labels, outputs, z_log_var, z_mean, input_shape)
    vae.add_loss(vae_loss)

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                # metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))
                metrics=tf.keras.metrics.CategoricalAccuracy())
    return encoder, decoder, vae


"""La pérdida en un VAE está compuesta por dos términos

1.   Pérdida de la reconstrucción: Esta mide cuanto se ajusta cada pixel de salida al valor de la etiqueta correspondiente.
2.   Término de regulación KL (Kullback–Leibler): Controla que el espacio latente esté formado un distribución normal multivariada centrada en cero.


"""


def reconstruction_loss(y_true, y_pred, z_log_var, z_mean, input_shape):
    # reconstruction_loss =  tf.keras.losses.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    reconstruction_loss = tf.keras.losses.categorical_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    reconstruction_loss /= input_shape[0] * input_shape[1] * input_shape[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = - K.mean(kl_loss, axis=-1)
    beta = 0.5
    vae_loss = K.mean(reconstruction_loss + beta * kl_loss)

    return vae_loss


"""A la hora de muestrear de las distribuciones normales que forman la huella latente es necesario utilizar un truco para que sea posible retropropagar el gradiente. Esto se debe a que muestrear de una distribución normal no es una operación derivable.

Con este fin, se introduce la variable "epsilon" que es un vector de valores que se oftienen muestreando de una distribución normal N(0, 1). El valor de epsilon se utiliza para hacer la selección de z mediante la ecuación: 

z = z_mean + sigma * epsilon 

donde  sigma = e^(z_var/2)
"""


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


"""Construimos el modelo de VAE.
Seleccionamos el número de canales en función del número de etiquetas.
"""

input_shape = (128, 128, 3)
out_channels = 3
latent_dim = 128
kernel_size = 3

encoder, decoder, vae = build_model(input_shape, out_channels, latent_dim, kernel_size)

"""Entrenamos el VAE.

En keras normalmente tendríamos que los datos de entrada son x_train y y_train se introducen como etiquetas tal que:

```
vae.fit(x_train, y_train, epochs...)
```

En este caso hemos definido una función de pérdida especial y se la hemos asignado al modelo de una forma "no estandar". Por esto, hemos creado los dos tensores de entrada "input" para x_train y "labels" para y_train. En esta situación tenemos que introducir los datos de la siguiente forma:

```
vae.fit([x_train, y_train], None, epochs...)
```

Donde `[x_train, y_train]` son las dos entradas a nuestro modelo de Keras y dado que nuestra función de pérdidas no utiliza la variable de entrada por defecto para las etiquetas de la función, en esta se introduce un "None".

Por último, hemos añadido una metrica de la forma convencional. Para que esta métrica funcione debemos introducir las etiquetas de forma convencional. Por lo tanto, introduciremos las etiquetas tanto en el tensor de entrada que hemos preparado para ello como en su variable dedicada de la función fit, es decir:

```
vae.fit([x_train, y_train], y_train, epochs...)
```

En el caso de los datos de validación hay que introducirlos siguiendo las mismas reglas:

```
validation_data=([x_val, y_val], y_val)
```

"""

batch_size = 128
epochs = 40

vae.fit([x_train, y_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=([x_val, y_val], y_val))

"""# Testing
En primer lugar vamos a pasar la red sobre el conjunto de test para ver si ha 
generalizado.

Para esto tenemos que cargar en primer lugar el conjunto de test.
"""

# testing_in = os.listdir(os.path.join(data_path, "original_testing"))
# testing_label = os.listdir(os.path.join(data_path, "ground_truth_testing"))
#
# # Ordenamos las imágenes por sequencia
# testing_in.sort(key=lambda x: (int(x.split("_")[0].split("l")[1]), int(x.split(".")[0].split("_")[1])))
# testing_label.sort(key=lambda x: (int(x.split("_")[0].split("t")[1]), int(x.split(".")[0].split("_")[1])))
# x_test = [plt_image.imread(os.path.join(data_path, "original_testing", name)) for name in testing_in[:100]]
# y_test = [plt_image.imread(os.path.join(data_path, "ground_truth_testing", name)) for name in testing_label[:100]]

"""Aplicaremos las mismas transformaciones que en el conjunto de entrenamiento, es decir, redimensionaremos laas imágenes hasta el tamaño de entrada de la red, daremos formato one hot a las etiquetas y normalizaremos los datos. """

# for i in range(x_test.shape[0]):
#     x_test[i] = cv2.resize(x_test[i], (128, 128))
#     y_test[i] = cv2.resize(y_test[i], (128, 128))
#
# # Normalizamos los datos de entrada en rango [0, 1]
# x_test = np.array(x_test) / 255.
# y_test = np.array(y_test) / 255.
# y_test = np.expand_dims(y_test[:, :, :, 1], axis=-1)
#
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)  # one hot encoding

"""la función evaluate realiza un predict sobre cada entrada del conjunto de test y extrae las métricas que hayamos indicado al compilar el modelo."""

vae.evaluate([x_test, y_test], y_test, batch_size=batch_size)

"""Por último, vamos a visualizar algunos de los resultados. 

La siguiente celda selecciona varias imágenes aleatorias y muestra su salida.
"""

figsize = 5
num_examples_to_generate = figsize * figsize
fig = plt.figure()
fig.suptitle('Resultados segmentación')
examples_index = np.random.choice(x_test.shape[0], figsize * 2)
examples = x_test[examples_index]
z_mean, z_log_var, z = encoder.predict(examples)
predictions = decoder.predict(z_mean)

plt.imshow(predictions[0, :, :, 0])
plt.show()
plt.imshow(predictions[0, :, :, 1])
plt.show()
plt.imshow(predictions[0, :, :, 2])
plt.show()

predictions = np.argmax(predictions, axis=-1)

for i in range(figsize * 2):
    plt.subplot(4, figsize, i + 1)
    plt.imshow(examples[i])
    plt.subplot(4, figsize, i + 1 + figsize * 2)
    plt.imshow(predictions[i], cmap='gray')

plt.axis('off')
plt.show()

"""Una característica interesande de los VAE es que su decodificador es un modelo generador. Podemos crear una huella latente y ajustar sus parámetros para obtener imágenes nuevas.

Por ejemplo con una red que segmente el tráfico, podríamos crear imagenes de segmentación de nuevas situaciones y a su vez utilizarlas para entrenar otro algoritmo en el manejo de un vehículo autónomo.

En la celda siguiente se muestrean varias huellas latented sobre una distribución normal para ver nuevos ejemplos.


"""

random_vector_for_generation = np.random.normal(size=(num_examples_to_generate, latent_dim))

predictions = decoder.predict(random_vector_for_generation)

predictions = np.argmax(predictions, axis=-1)
fig = plt.figure(figsize=(figsize, figsize))
fig.suptitle('Nuevos ejemplos generados')
for i in range(num_examples_to_generate):
    plt.subplot(figsize, figsize, i + 1)
    plt.imshow(predictions[i])
plt.axis('off')