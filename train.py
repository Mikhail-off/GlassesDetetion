import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 8
IMAGE_SIZE = 224
SEED = 42
IMAGE_COUNT = 1000#47917

DATASET_PATH = 'C:\\GlassesDetetion\\data\\'
MODEL_NAME = 'model.hdf5'

def normalize_image(img_as_array):
    return (img_as_array - 127.5) / 127.5


def train_generator(data_dir):
    return ImageDataGenerator(preprocessing_function=normalize_image).flow_from_directory(
        data_dir, batch_size=BATCH_SIZE,
        class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb", seed=SEED)


def build_model():
    backbone = MobileNetV2((224, 224, 3), alpha=0.5, include_top=False)

    cur = backbone.output
    cur = GlobalAveragePooling2D()(cur)
    cur = Dense(1, activation='sigmoid')(cur)

    model = Model(backbone.input, cur)
    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def train_model(train_data_path):
    train_data_generator = train_generator(train_data_path)

    #model = load_model('model.hdf5')
    model = build_model()
    model.fit_generator(train_data_generator, steps_per_epoch=IMAGE_COUNT, epochs=1)
    model.save(MODEL_NAME)
    return model


if __name__ == '__main__':
    train_model(DATASET_PATH)