import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt
import cv2

import keras
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, LeakyReLU
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--train_dataset", "-train", type=str, required=True,
                       help="Путь до папки с датасетом")
argparser.add_argument("--test_dataset", "-test", type=str, required=True,
                       help="Путь до папки с датасетом для обучения")
argparser.add_argument("--n_classes", type=int, required=True,
                       help="Сколько классов в датаете")
argparser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                       help="Learning rate во время обучения")
argparser.add_argument("--continue_from_model", "-cfm", type=str, default=None,
                       help="Загрузить модель для дообучения")
argparser.add_argument("--batch_size", "-bs", type=int, default=8,
                       help="Batch size для обучения")
argparser.add_argument("--epochs", "-e", type=int, default=1,
                       help="Кол-во эпох обучения")
argparser.add_argument("--logs_dir", "-l", type=str, default=None,
                       help="Путь до папки с логами")
argparser.add_argument("--visualize", "-vis", action='store_true',
                       help="Нужно ли сохранить аугментированные изображения")

args = argparser.parse_args()

TRAIN_DATASET_PATH = args.train_dataset
TEST_DATASET_PATH = args.test_dataset
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
N_CLASSES = args.n_classes
LR = args.learning_rate
LOGS_DIR = args.logs_dir
VISUALIZE = args.visualize
LOADING_MODEL = args.continue_from_model

MODEL_NAME = 'model.hdf5'
IMAGE_SIZE = 224
SEED = 42


def normalize_image(img_as_array):
    return (img_as_array - 127.5) / 127.5


def train_generator(data_dir):
    if VISUALIZE:
        save_dir = os.path.join(LOGS_DIR, 'augmented_data')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    return ImageDataGenerator(preprocessing_function=normalize_image,
                              horizontal_flip=True, brightness_range=(0.5, 1.5),
                              zoom_range=(0.8, 1.2), rotation_range=20).flow_from_directory(
        data_dir, batch_size=BATCH_SIZE,
        class_mode='sparse', target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb", seed=SEED,
        save_to_dir=save_dir, shuffle=True)


def test_generator(data_dir):
    return ImageDataGenerator(preprocessing_function=normalize_image).flow_from_directory(
        data_dir, batch_size=BATCH_SIZE,
        class_mode='sparse', target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb", seed=SEED)


def build_model():
    backbone = MobileNetV2((IMAGE_SIZE, IMAGE_SIZE, 3), alpha=0.35, include_top=False)
    #backbone.trainable = False

    cur = backbone.output
    cur = GlobalAveragePooling2D()(cur)
    cur = Dense(128, activation=None)(cur)
    cur = LeakyReLU(0.2)(cur)
    cur = Dropout(0.2)(cur)
    if N_CLASSES == 1 or N_CLASSES == 2:
        act_f = 'sigmoid'
        loss_f = 'binary_crossentropy'
    else:
        assert N_CLASSES > 1
        act_f = 'softmax'
        loss_f = 'sparse_categorical_crossentropy'

    cur = Dense(N_CLASSES, activation=act_f)(cur)

    model = Model(backbone.input, cur)
    model.compile(optimizer=Adam(lr=LR), loss=loss_f, metrics=['accuracy'])
    return model


def train_model(train_data_path, test_data_path, model):
    model.summary()
    train_data_generator = train_generator(train_data_path)
    test_data_generator = test_generator(test_data_path)

    model.fit_generator(train_data_generator, steps_per_epoch=train_data_generator.samples // BATCH_SIZE, epochs=EPOCHS,
                        validation_data=test_data_generator, validation_steps=test_data_generator.samples // BATCH_SIZE)
    model.save(MODEL_NAME)


if __name__ == '__main__':
    os.makedirs(LOGS_DIR, exist_ok=True)
    model = build_model()
    if args.continue_from_model is not None:
        model.load_weights(LOADING_MODEL)

    train_model(TRAIN_DATASET_PATH, TEST_DATASET_PATH, model)
