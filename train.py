import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt
import cv2

import keras
import tensorflow as tf
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from random import randint

EPOCH_COUNT = 5
BATCH_SIZE = 16
IMAGE_SIZE = 224
SEED = randint(0, 1000)
IMAGE_COUNT = 3804#47917
N_CLASSES = 3
LR = 1e-5

DATASET_PATH = 'C:\\GlassesDetetion\\global_data\\train\\'
MODEL_NAME = 'model.hdf5'


def normalize_image(img_as_array):
    return (img_as_array - 127.5) / 127.5


def train_generator(data_dir):
    return ImageDataGenerator(preprocessing_function=normalize_image, rotation_range=10,
                              horizontal_flip=True,
                              width_shift_range=[-30, 30], height_shift_range=[-30, 30],
                              brightness_range=[0.6, 1.4], zoom_range=[0.8, 1.2]).flow_from_directory(
        data_dir, batch_size=BATCH_SIZE,
        class_mode='categorical', target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb", seed=SEED, shuffle=True)


def build_model():
    backbone = MobileNetV2((224, 224, 3), alpha=0.5, include_top=False)

    cur = backbone.output
    cur = Conv2D(128, kernel_size=(1, 1), activation='relu')(cur)
    cur = GlobalAveragePooling2D()(cur)

    cur = Dense(N_CLASSES, activation='softmax')(cur)
    loss_name = 'categorical_crossentropy'

    model = Model(backbone.input, cur)

    model.compile(optimizer=Adam(learning_rate=LR), loss=loss_name, metrics=['accuracy'])
    model.summary()

    return model


def train_model(train_data_path):
    train_data_generator = train_generator(train_data_path)

    model = load_model('model.hdf5')
    #model = build_model()
    model.fit_generator(train_data_generator, steps_per_epoch=IMAGE_COUNT // BATCH_SIZE, epochs=EPOCH_COUNT,
                        class_weight={0: 1., 1: 1., 2: 2.})
    model.save(MODEL_NAME)
    return model


if __name__ == '__main__':
    train_model(DATASET_PATH)