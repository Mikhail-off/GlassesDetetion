import pandas as pd
import numpy as np
import os
from shutil import copyfile
from tqdm import tqdm

RAW_DATA_PATH = 'C:\\GlassesDetetion\\raw_data\\'
DATA_PATH = 'C:\\GlassesDetetion\\data\\'
RAW_IMAGES_PATH = os.path.join(RAW_DATA_PATH, 'images')
MARKUP_FILE_PATH = os.path.join(RAW_DATA_PATH, 'markup.csv')

NAME_COL = 'name'
CLASS_COL = 'class'

def copy_image_to_folder(image_name, image_class):
    image_class = str(image_class)
    image_class_dir = os.path.join(DATA_PATH, image_class)
    os.makedirs(image_class_dir, exist_ok=True)
    copyfile(os.path.join(RAW_IMAGES_PATH, image_name), os.path.join(image_class_dir, image_name))

def main():
    markup_df = pd.read_csv(MARKUP_FILE_PATH)
    for image_name, image_class in tqdm(markup_df.values):
        copy_image_to_folder(image_name, image_class)

if __name__ == '__main__':
    main()