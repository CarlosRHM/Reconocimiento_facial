# La idea de la version 2 es que al no poder entrenar con el conjunto completo
# los datos y etiquetas de CelebA, se toman de la misma base de datos, un
# numero menor de imagenes y etiquetas.
# La nueva base de datos tendrá 50K imagenes y 20 etiquetas

import numpy as np
import pandas as pd
import patlib
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import wandb
from wandb.keras import WandbCallback
#-------------------------------------------------------------------------------
ih = 128
iw = 128
N_d = 50000
N_e = 20

percent = 0.3

BATCH_SIZE = 64
epochs = 30

STEPS_PER_EPOCH =  (N_d*(1-percent)) // BATCH_SIZE
VALIDATION_STEPS = (N_d * percent) // BATCH_SIZE
#-------------------------------------------------------------------------------
#                            Carga de Datos

np.set_printoptions(precision=4)

df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header = None)
print('=======================================================================')
print('CelebA')
print(df.head())
print('=======================================================================')

# NOTE: Tomando los 20 atributos mas importantes para reconocer una cara
#       se definen en la matriz attri_20
#       Se toman los atributos de en attributes_index y se convierten los -1
#       en 0 para usar el 'binary_crossentropy' como costo y las funciones
#       que se definen de 0 a 1 como sigmoide.
attri_20 = [1,3,6,7,8,11,13,17,20,21,22,23,24,25,27,29,31,32,33,39]
attri_20 = attri_20[0:N_e]
attributes_index = df.iloc[0:N_d,attri_20].to_numpy()
for i in range(N_d):
    for j in range(N_e):
        if attributes_index[i,j] == -1:
            attributes_index[i,j] = 0
files = tf.data.Dataset.from_tensor_slices(df.iloc[0:N_d,0])
attributes = tf.data.Dataset.from_tensor_slices(attributes_index)

data = tf.data.Dataset.zip((files, attributes))

path_to_images = 'Data/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [ih,iw])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)
#dataset = labeled_images.repeat().batch(BATCH_SIZE)


print('=======================================================================')

print('labeled_images:')
print(labeled_images)
print('labeled_images_len:')
print(len(labeled_images))

#print('Dataset:')
#print(dataset)

print('=======================================================================')

#for image, attri in labeled_images.take(3):
#    print(attri)
#    plt.imshow(image)
#    plt.show()
#exit()

ds_size = int(N_d * (1-percent))

training_data = labeled_images.take(ds_size)
#print(len(training_data))
training_data = training_data.repeat().batch(BATCH_SIZE)

test_data = labeled_images.skip(ds_size)
#print(len(test_data))
test_data = test_data.repeat().batch(BATCH_SIZE)

print('=======================================================================')
print('split dataset in training and test data')

print('training_data')
print(training_data)

print('test_data')
print(test_data)

print('=======================================================================')

#exit()

#-------------------------------------------------------------------------------
#                               WandB callbaks
wandb.init(project="Reconocimiento_Facial_v2.8")
wandb.config.epochs = epochs
wandb.config.batch_size = BATCH_SIZE
wandb.config.optimizer = "rmsprop"
#-------------------------------------------------------------------------------
#                           Estructura de la red
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(ih,iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(N_e))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["binary_accuracy"])
model.summary()
#exit()

history = model.fit(
                training_data,
                epochs = epochs,
                steps_per_epoch=STEPS_PER_EPOCH,
                batch_size = BATCH_SIZE,
                validation_data = test_data,
                validation_steps = VALIDATION_STEPS,
                callbacks = [WandbCallback()]
                )
model.save('redes_entrenadas/reconocimiento_facial_v2.8.h5')
