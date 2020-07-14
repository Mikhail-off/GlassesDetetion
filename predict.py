from skimage.io import imread
from skimage.transform import resize
from keras.models import load_model
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import cv2
from cv2 import cvtColor

argparser = argparse.ArgumentParser()
argparser.add_argument("--src_images", "-src", type=str,
                       help="Путь до папки с изображениями")
argparser.add_argument("--dst_file", "-dst", type=str,
                       help="Путь до файла с результатами")

args = argparser.parse_args()

IMAGES_PATH = args.src_images
RESULT_FILE_PATH = args.dst_file
MODEL_NAME = 'model.hdf5'
INPUT_SHAPE = (224, 224, 3)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray_image = cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = image[y: y + h, x: x + w, ::-1]
        cropped_faces.append(face)
    return cropped_faces

def classify_face(image, model):
    image = resize(image, output_shape=(INPUT_SHAPE))
    image = np.array([image])
    image_class = int(model.predict(image)[0] > 0.5)
    return image_class

def main():
    result = []

    model = load_model(MODEL_NAME)
    for image_name in tqdm(os.listdir(IMAGES_PATH)):
        image = cv2.imread(os.path.join(IMAGES_PATH, image_name))
        faces = detect_faces(image)
        if len(faces) == 1:
            image_class = classify_face(faces[0], model)
        else:
            image_class = 2
        result.append([image_name, image_class])

    result = pd.DataFrame(data=result, columns=['name', 'class'])
    result.to_csv(RESULT_FILE_PATH, index=False)

if __name__ == '__main__':
    main()