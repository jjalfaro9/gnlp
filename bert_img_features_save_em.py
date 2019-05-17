import pandas as pd
import numpy as np
import pickle

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

def _read_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
def _save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
def _get_vgg16_fts(image_directory, image_paths, vgg):
    print("Get vgg16 fts...")
    image_objects = [image.load_img(f'{image_directory}/{img_path}', target_size=(224, 224)) for img_path in image_paths]
    x_images = [image.img_to_array(x) for x in image_objects]
    x_images_exp = [np.expand_dims(img_data, axis=0) for img_dat in x_images]
    imgs = [preprocess_input(img_d) for img_d in x_images_exp]
    img_fts = vgg.predict(imgs)
    return [np.array(vg_ft).flatten() for vg_ft in img_fts]

def _save_img_bert_fts(image_directory):
    # read in icons for training and testing 
    # get those features for each 
    # save them into a file
    print("Load files...")
    img_icons_train = _read_file('./img_icons_train.p')
    img_icons_test = _read_file('./img_icons_test.p')

    vgg = VGG16(weights='imagenet', include_top=False)
    _save_file('img_fts_train.p', _get_vgg16_fts(image_directory, img_icons_train, vgg))
    _save_file('img_fts_test.p', _get_vgg16_fts(image_directory, img_icons_test, vgg))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("image_directory")
args = parser.parse_args()

if __name__ == '__main__':
    _save_img_bert_fts(args.image_directory)
	
