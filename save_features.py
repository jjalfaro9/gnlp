import json
import os
import pickle
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

parser = argparse.ArgumentParser()
parser.add_argument("PNG_DIR")
parser.add_argument("metadata_path")
args = parser.parse_args()


def _read_in_data(metadata_path, PNG_DIR):
    data = []
    unique_pngs = {}
    counter = 0
    errors = 0
    dups = 0

    # create better categories for installs count
    dlcounts = [         0,          1,          5,         10,         50,
              100,        500,       1000,       5000,      10000,
            50000,     100000,     500000,    1000000,    5000000,
         10000000,   50000000,  100000000,  500000000, 1000000000]
    dlcounts_nums = {}
    dlc = {}
    for i,x in enumerate(dlcounts):
        dlc[x] = i
        dlcounts_nums[i] = 0

    with open(metadata_path, 'r') as metadata_file:
        lines = metadata_file.readlines()
    import random
    random.shuffle(lines)

    for line in lines:
        app_result = {}
        app_data = json.loads(line)
        app_id = app_data['appId']
        icon_path = f"{PNG_DIR}/{app_id}.png"
    #         print(icon_path)
        counter += 1
        if (counter % 5000 == 0):
            print(counter, app_id)
        if os.path.isfile(icon_path):
            if (icon_path in  unique_pngs):
                dups += 1
                continue
            unique_pngs[icon_path] = icon_path
            try:
                app_result['icon'] = f'{app_id}.png'
                app_result['appId'] = app_data['appId']
                app_result['contentRating'] = app_data['contentRating']
                app_result['description'] = app_data['description']
                app_result['descriptionHTML'] = app_data['descriptionHTML']
                app_result['genre'] = app_data['genre']
                app_result['genreId'] = app_data['genreId']
                app_result['histogram'] = app_data['histogram']
                app_result['installs'] = app_data['installs']
                app_result['minInstalls'] = app_data['minInstalls']
                app_result['price'] = app_data['price']
                app_result['priceText'] = app_data['priceText']
                app_result['ratings'] = app_data['ratings']
                app_result['reviews'] = app_data['reviews']
                app_result['score'] = app_data['score']
                app_result['scoreText'] = app_data['scoreText']
                app_result['size'] = app_data['size']
                app_result['summary'] = app_data['summary']
                app_result['title'] = app_data['title']
                app_result['updated'] = app_data['updated']
                app_install_num = dlc[app_data['minInstalls']]
                app_result['installsCategory'] = app_install_num
                if dlcounts_nums[app_install_num] > 10000:
                    continue
                dlcounts_nums[app_install_num] += 1
                app_result['reviewsCategory'] = len(str(app_data['reviews']))
                data.append(app_result)

            except KeyError as ke:
                errors+= 1
                continue
        else:
            continue
    print(counter, errors, dups)
    return pd.DataFrame(data)

def _split_corpus_test_train(data):
    descriptions = data['description'].to_numpy()
    icons = data['icon'].to_numpy()
    # these lengths should be the same!
    # want to use them to compute the percentage for the training set
    # want 110k training, 10k testing (or around there)
    # usually the number is 112,282 and the magic number that gets us around there is .9 => 101k, 11k

    descriptions_len, icons_len = len(descriptions), len(icons)
    print("Lengths should be the same, are they and are they equal to 11282? (%s) (%s)" % (descriptions_len, icons_len))
    if descriptions_len == 112282:
        ts = 0.9
    else:
        # neeed to compute ts => x / len = ts , 110k / len = (?) but what if len is not long enough
        if descriptions_len > 100000:
            ts = 100000 / descriptions_len
        else:
            ts = 0.8
    return train_test_split(descriptions, icons, train_size=ts)

def _get_features(pngs_dir, training, testing):
    model = VGG16(weights='imagenet', include_top=False)
    training_features = _extract_features(pngs_dir, training, model)
    testing_features = _extract_features(pngs_dir, testing, model)
    return (training_features, testing_features)

def _extract_features(pngs_dir, icons, model):
    vgg16_features = []
    counter = 0
    for i in icons:
        counter += 1
        if (counter % 5000 == 0):
            print("extracting features: =>", counter, pngs_dir + i)
        img = image.load_img(pngs_dir + i, target_size=(224,224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        vgg16_features.append(np.array(vgg16_feature).flatten())
    return vgg16_features

def _save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def save_features(metadata_path, png_dir):
    data = _read_in_data(metadata_path, png_dir)
    d_train, d_test, i_train, i_test = _split_corpus_test_train(data)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    bow_train = vectorizer.fit_transform(d_train)
    bow_test = vectorizer.transform(d_test)

    img_features_train, img_features_test = _get_features(png_dir, i_train, i_test)

    _save_file('description_train.p', d_train)
    _save_file('description_test.p', d_test)
    _save_file('icon_train.p', i_train)
    _save_file('icon_test.p', i_test)
    _save_file('bow_fts_train.p', bow_train)
    _save_file('bow_fts_test.p', bow_test)
    _save_file('img_fts_train.p', img_features_train)
    _save_file('img_fts_test.p', img_features_test)

if __name__ == '__main__':
	save_features(args.metadata_path, args.PNG_DIR)
