import tensorflow as tf

import patlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

"""
# NOTE: prepara el archivo de atributos colocando un solo espacio
with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())
    print("skipping headers : " + f.readline())
    with open('list_attr_celeba_prepared.txt', 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            newf.write(new_line)
            newf.write('\n')
"""

df = pd.read_csv('list_attr_celeba_prepared.txt', sep=' ', header = None)
#print(df.head())
#exit()

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))
print(data)
exit()

path_to_images = 'Data/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)
print(labeled_images)

for image, attri in labeled_images.take(1):
    plt.imshow(image)
    plt.show()
