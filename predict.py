from skimage.io import imread
from skimage.transform import resize
from keras.models import load_model
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd


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


def main():
    result = []

    model = load_model(MODEL_NAME)
    for i, image_name in tqdm(enumerate(os.listdir(IMAGES_PATH))):
        image = imread(os.path.join(IMAGES_PATH, image_name))
        image = resize(image, output_shape=(INPUT_SHAPE))
        image = np.array([image])
        image_class = int(model.predict(image)[0] > 0.5)
        result.append([image_name, image_class])
        if i == 100:
            break
    result = pd.DataFrame(data=result, columns=['name', 'class'])
    result.to_csv(RESULT_FILE_PATH, index=False)

if __name__ == '__main__':
    main()