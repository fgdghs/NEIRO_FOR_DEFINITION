import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib

import numpy as np
import tensorflow as tf

import keras
from keras import layers
from keras import Sequential

## Путь до папки с датасетом раздутым
dataset_dir = pathlib.Path('F:\\PEDOVKI_DATASET_BIG')
batch_size = 32
img_width = 180
img_height = 180

#path_to_img - путь к изображению
#path_to_model_weight - путь к обученной нейросети
#Функция возвращает liked или disliked
def pedovka_definition(path_to_img, path_to_model_weight):
    
    train_ds = keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

    val_ds = keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

    num_classes = len(train_ds.class_names)

    model = Sequential([

    keras.layers.Rescaling(scale=1/255,input_shape=(img_height,img_width,3)),

    keras.layers.RandomFlip('horizontal', input_shape=(img_height,img_width,3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.2),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),

    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(
    optimizer='adam',
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
    )

    
    model.load_weights(path_to_model_weight)

    img = keras.utils.load_img(path_to_img, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return class_names[np.argmax(score)]

