
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance

import pandas as pd
import numpy as np

import os
import json

def get_data_frame(metadata_path, PNG_DIR):
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


def get_features(pngs_dir, training, testing):
    model = VGG16(weights='imagenet')
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


def cos_cdist(vector, matrix):
        v = vector.reshape(1, -1)
        return distance.cdist(matrix, v, 'cosine').reshape(-1)
