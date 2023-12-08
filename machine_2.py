import os
import keras

#if using Theano with GPU
#os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

import random
import numpy as np
import pandas as pd
from PIL import Image


# Configurações
root = r'D:\DIO_DataScience\test'

print(root)

exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces', 'Test/.ipynb_checkpoints']
train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(root) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

print(categories)

def get_image(path, target_size=(224, 224)):
    try:
        img = Image.open(path)
        img = img.resize(target_size)
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        return img, x
    except Exception as e:
        print(f"Erro ao processar a imagem {path}: {str(e)}")
        return None, None


data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        if img is not None and x is not None:
            data.append({'x': x.shape[:3], 'y': c}) #ajuste shape[:3}]
num_classes = len(categories)
print(num_classes)  #resultado esperado 2 (duas classes Gatos e Cachorros)

random.shuffle(data)
train_split, val_split = 0.7, 0.15

idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]


#*******************************************************************
x_train = np.array([t["x"] for t in train ])
print("Shape x_train:", x_train.shape)
y_train = np.array([t["y"] for t in train ])
print("Shape y_train:", y_train)

x_val = np.array([t["x"] for t in val ])
print("Shape x_val:", x_val)
y_val = np.array([t["y"] for t in val ])
print("Shape y_val:", y_val)

x_test = np.array([t["x"] for t in test ])
print("Shape x_test", x_test)
y_test = np.array([t["y"] for t in test ])
print("Shape y_test", y_test)



# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

# summary
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)


images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)
plt.show()