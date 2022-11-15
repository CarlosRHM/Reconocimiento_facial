import os
import numpy as np
import tensorflow as tf

import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop

import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt

ih = 128
iw = 128

percent = 0.15

BATCH_SIZE = 32
epochs = 20

N_d = 600

STEPS_PER_EPOCH =  (2*N_d*(1-percent)) // BATCH_SIZE
VALIDATION_STEPS = (2*N_d * percent) // BATCH_SIZE
#==============================================================================
#                               procesamiento de Datos
fotos_mias = "Data/fotos_mias/"
CelebA = "Data/img_align_celeba/"

fotos_mias_files = os.listdir(fotos_mias)
CelebA_files = os.listdir(CelebA)

fotos_mias_attri = np.ones(N_d, dtype = "int")
CelebA_attri = np.zeros(N_d, dtype = "int")

fotos_mias_files = tf.data.Dataset.from_tensor_slices(fotos_mias_files)
fotos_mias_attri = tf.data.Dataset.from_tensor_slices(fotos_mias_attri)
CelebA_files = tf.data.Dataset.from_tensor_slices(CelebA_files)
CelebA_attri = tf.data.Dataset.from_tensor_slices(CelebA_attri)

fotos_mias_data = tf.data.Dataset.zip((fotos_mias_files,fotos_mias_attri))
CelebA_data = tf.data.Dataset.zip((CelebA_files,CelebA_attri))

def process_file_fotos_mias(file_name, fotos_mias_attri):
    image = tf.io.read_file(fotos_mias + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [ih,iw])
    image /= 255.0
    return image, fotos_mias_attri

def process_file_CelebA(file_name, CelebA_attri):
    image = tf.io.read_file(CelebA + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [ih,iw])
    image /= 255.0
    return image, CelebA_attri

labeled_images_fotos_mias = fotos_mias_data.map(process_file_fotos_mias)
labeled_images_CelebA = CelebA_data.map(process_file_CelebA)

labeled_images = labeled_images_fotos_mias.concatenate(labeled_images_CelebA)
labeled_images = labeled_images.shuffle(N_d*2)

print('=======================================================================')

print('labeled_images:')
print(labeled_images)
print('labeled_images_len:')
print(len(labeled_images))

print('=======================================================================')

dataset_size = int(N_d*2 * (1-percent))

training_data = labeled_images.take(dataset_size)
training_data = training_data.repeat().batch(BATCH_SIZE)

test_data = labeled_images.skip(dataset_size)
test_data = test_data.repeat().batch(BATCH_SIZE)

print('=======================================================================')
print('split dataset in training and test data')

print('training_data')
print(training_data)

print('test_data')
print(test_data)

print('=======================================================================')
#==============================================================================

wandb.init(project="Reconocimiento_Facial_Usuario")
wandb.config.epochs = epochs
wandb.config.batch_size = BATCH_SIZE
wandb.config.optimizer = "adam"

pre_trained_model = tf.keras.models.load_model(
                            'redes_entrenadas/reconocimiento_facial_v2.8.h5')
#pre_trained_model.summary()
#exit()

model = Sequential()

for layer in pre_trained_model.layers[:9]:
    model.add(layer)

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))

for layer in model.layers[:9]:
    layer.trainable = False

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["binary_accuracy"])
model.summary()

history = model.fit(
                training_data,
                epochs = epochs,
                steps_per_epoch=STEPS_PER_EPOCH,
                batch_size = BATCH_SIZE,
                validation_data = test_data,
                validation_steps = VALIDATION_STEPS,
                callbacks = [WandbCallback()]
                )

model.save("redes_entrenadas/reconocimiento_facial_Usuario_v4.h5")
