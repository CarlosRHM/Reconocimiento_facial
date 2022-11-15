import numpy as np
import pandas as pd
import patlib
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, Adam

#-------------------------------------------------------------------------------
ih = 100
iw = 100
BATCH_SIZE = 32
STEPS_PER_EPOCH =  202599 // BATCH_SIZE
epochs = 30

#-------------------------------------------------------------------------------
#                            Carga de Datos

np.set_printoptions(precision=4)

df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header = None)
print('=======================================================================')
print('CelebA')
print(df.head())
print('=======================================================================')

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())

data = tf.data.Dataset.zip((files, attributes))
#print(data)

path_to_images = 'Data/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [ih,iw])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)
dataset = labeled_images.repeat().batch(BATCH_SIZE)


print('=======================================================================')

print('labeled_images:')
print(labeled_images)
print('labeled_images_len:')
print(len(labeled_images))

print('Dataset:')
print(dataset)

print('=======================================================================')

#for image, attri in labeled_images.take(1):
#    plt.imshow(image)
#    plt.show()
#exit()

#-------------------------------------------------------------------------------
#                           Estructura de la red
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(ih,iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Activation('tanh'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
model.summary()
#exit()

history = model.fit(
                dataset,
                epochs = epochs,
                steps_per_epoch=STEPS_PER_EPOCH,
                batch_size = BATCH_SIZE
                )
model.save('reconocimiento_facial.h5')
