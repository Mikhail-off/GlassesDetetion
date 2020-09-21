from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from keras.models import load_model
import os
from sklearn.metrics import classification_report

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--images_path", "-ip", type=str, required=True,
                       help="Путь до папки с изображениями")
argparser.add_argument("--markup_path", "-mp", type=str, default=None,
                       help="Путь до папки с разметкой. Если не указано, то метрики не считаются")
argparser.add_argument("--model_path", "-m", type=str, required=True,
                       help="Путь до тестируемой модели")
argparser.add_argument("--result_path", "-rp", type=str, required=True,
                       help="Путь до результирующей разметки")
argparser.add_argument("--metrics", action="store_true",
                       help="Нужно ли считать метрики")

# TODO
"""
argparser.add_argument("--logs", "-l", type=str, default=None,
                       help="Путь до папки, куда складываются визуализированные результаты")
"""

args = argparser.parse_args()

IMAGES_PATH = args.images_path
MARKUP_PATH = args.markup_path
MODEL_PATH = args.model_path
RESULT_PATH = args.result_path
METRICS = args.metrics
# TODO
#LOG_DIR = args.logs

IMAGE_SIZE = (224, 224)
CLASS_COL = 'Class'
CROP_COLS = ['TopLeftX', 'TopLeftY', 'BottomRightX', 'BottomRightY']
MARKUP_EXT = '.csv'


def open_image(image_name):
    return Image.open(os.path.join(IMAGES_PATH, image_name)).convert("RGB")


def preprocess_faces(faces):
    for i, face in enumerate(faces):
        faces[i] = np.array(face.resize(IMAGE_SIZE))
    faces = np.array(faces).astype(float)
    return (faces - 127.5) / 127.5


def crop_faces(image, face_coords):
    faces = []
    for face_coord in face_coords:
        faces.append(image.crop(face_coord))
    return faces


def is_image(file_path):
    return os.path.splitext(file_path)[-1] in ['.jpg', '.png', '.JPG', '.JPEG']


def main():
    os.makedirs(RESULT_PATH, exist_ok=True)
    image_names = list(filter(is_image, os.listdir(IMAGES_PATH)))
    markup_names = [os.path.splitext(image_name)[0] + MARKUP_EXT
                    for image_name in image_names]

    model = load_model(MODEL_PATH)

    y_true = []
    y_pred = []

    for image_name, markup_name in tqdm(zip(image_names, markup_names), total=len(image_names)):
        markup_df = pd.read_csv(os.path.join(MARKUP_PATH, markup_name))
        image = open_image(image_name)
        faces = crop_faces(image, markup_df[CROP_COLS].values)
        faces = preprocess_faces(faces)
        faces_classes = np.argmax(model.predict(faces), axis=-1)

        if METRICS:
            y_pred += list(faces_classes)
            y_true += list(markup_df[CLASS_COL].values)

        markup_df[CLASS_COL] = faces_classes
        markup_df.to_csv(os.path.join(RESULT_PATH, markup_name), index=False)

    if METRICS:
        print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    main()