#!/usr/bin/python3
# -*- coding: utf-8 -*-
from skimage.transform import resize
from keras.models import load_model
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import cv2
from cv2 import cvtColor
from time import time

argparser = argparse.ArgumentParser()
argparser.add_argument("--src_images", "-src", type=str,
                       help="Путь до папки с изображениями")
argparser.add_argument("--dst_file", "-dst", type=str,
                       help="Путь до файла с результатами")
argparser.add_argument("--threshold", "-th", type=float,
                       help="Минимальная уверенность", default=0.0)


args = argparser.parse_args()

IMAGES_PATH = args.src_images
RESULT_FILE_PATH = args.dst_file
MODEL_NAME = 'model.hdf5'
INPUT_SHAPE = (224, 224, 3)
THRESHOLD = args.threshold

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(image, with_opencv=True):
    if not(with_opencv):
        return [image]
    gray_image = cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = image[y: y + h, x: x + w, :]
        cropped_faces.append(face)
    return cropped_faces


def classify_face(image, model):
    image = cv2.resize(image, INPUT_SHAPE[:2])
    image = np.array([image]).astype(float)
    image = (image - 127.5) / 127.5
    image = image[:, :, ::-1]
    probs = model.predict([image])[0]
    image_class = np.argmax(probs)
    if probs[image_class] < THRESHOLD:
        image_class = 0
    return image_class


def main():
    result = []

    model = load_model(MODEL_NAME)
    model.predict(np.zeros((1,) + INPUT_SHAPE))
    
    avg_time_per_image = 0
    image_count = 0
    
    image_names = os.listdir(IMAGES_PATH)
    for image_name in tqdm(image_names):
        time_spent = time()
        image_path = os.path.join(IMAGES_PATH, image_name)  
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_count += 1
        
        faces = detect_faces(image, with_opencv=False)
        if len(faces) == 1:
            image_class = classify_face(faces[0], model)
        else:
            image_class = -1
        time_spent = time() - time_spent
        result.append([image_name, image_class])
        avg_time_per_image += time_spent
    assert image_count != 0
    avg_time_per_image /= image_count
    
    print('Average time spent on one image is:', avg_time_per_image)
    
    result = pd.DataFrame(data=result, columns=['name', 'class'])
    result.to_csv(RESULT_FILE_PATH, index=False)


if __name__ == '__main__':
    main()