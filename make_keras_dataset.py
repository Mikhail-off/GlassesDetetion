import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--src_dataset", "-src", type=str,
                       help="Путь до папки с датасетом")
argparser.add_argument("--dst_dataset", "-dst", type=str,
                       help="Путь до папки с датасетом для обучения")
argparser.add_argument("--train_rate", "-tr", type=float, default=True,
                       help="Процент изображений для обучения")
argparser.add_argument("--crop", action='store_true',
                       help="Нужно ли обрезать картинку")
argparser.add_argument("--n_classes", type=int,
                       help="Сколько классов в датаете")

args = argparser.parse_args()

RAW_DATASET_PATH = args.src_dataset
TRAIN_DATASET_PATH = args.dst_dataset
NEED_CROP = args.crop
TRAIN_RATE = args.train_rate
N_CLASSES = args.n_classes

CLASS_COL = 'Class'
CROP_COLS = ['TopLeftX', 'TopLeftY', 'BottomRightX', 'BottomRightY']
MARKUP_EXT = '.csv'
DST_IMAGE_EXT = '.png'


def is_image(file_path):
    return os.path.splitext(file_path)[-1] in ['.jpg', '.png', '.JPG', '.JPEG']


def form_dataset(image_names, dst_dataset_folder):
    for i in range(N_CLASSES):
        os.makedirs(os.path.join(dst_dataset_folder, str(i)), exist_ok=True)

    obj_ind = 0
    for image_name in tqdm(image_names):
        image_name, orig_image_ext = os.path.splitext(image_name)
        markup_name = image_name + MARKUP_EXT
        df = pd.read_csv(os.path.join(RAW_DATASET_PATH, 'Markup', markup_name))
        for i in range(len(df)):
            obj_class = df[CLASS_COL].values[i]
            image = Image.open(os.path.join(RAW_DATASET_PATH, 'Image', image_name + orig_image_ext)).convert('RGB')
            if NEED_CROP:
                image = image.crop(box=df[CROP_COLS].values[i])

            obj_ind += 1
            dst_image_path = os.path.join(dst_dataset_folder, str(obj_class), '%06d' % obj_ind + DST_IMAGE_EXT)
            image.save(dst_image_path)


def main():
    image_names = np.array(list(filter(is_image, os.listdir(os.path.join(RAW_DATASET_PATH, 'Image')))))
    np.random.shuffle(image_names)

    train_left = 0
    train_right = int(len(image_names) * TRAIN_RATE)
    train_image_names = image_names[train_left:train_right]

    test_left = train_right
    test_right = test_left + int(len(image_names) * (1 - TRAIN_RATE) / 2)
    test_image_names = image_names[test_left:test_right]

    val_left = test_right
    val_right = len(image_names)
    val_image_names = image_names[val_left:val_right]

    for name, cur_image_names in zip(['Train', 'Test', 'Validation'],
                                     [train_image_names, test_image_names, val_image_names]):
        dataset_path = os.path.join(TRAIN_DATASET_PATH, name)
        form_dataset(cur_image_names, dataset_path)


if __name__ == '__main__':
    main()
