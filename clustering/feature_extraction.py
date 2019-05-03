from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument("PNG_DIR", help="directory for all images")
parser.add_argument("metadata_path", help="file to all data")
args = parser.parse_args()

def get_data_frame():
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

    with open(args.metadata_path, 'r') as metadata_file:
        lines = metadata_file.readlines()
    import random
    random.shuffle(lines)

    for line in lines:
        app_result = {}
        app_data = json.loads(line)
        app_id = app_data['appId']
        icon_path = f"{args.PNG_DIR}/{app_id}.png"
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
# the only problem with this is that there is a lot of data, so idk if this will blow up
def get_img_features(data_frame):
    model = VGG16(weights='imagenet')
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        preprocessing_function=preprocess_input
    )
    img_gen = data_gen.flow_from_dataframe(
        data_frame[:],
        directory=PNG_DIR,
        x_col="icon",
        class_model="None",
        target_size=(224,224),
        batch_size=32,
        shuffle=False
    )

    return model.predict_generator(
        generator=img_gen
    )


def cluster_imgs(imgFeatures):
    kmeans = KMeans()
    param_grid = {"n_clusters": range(2, 110)} #change this range, lol idk what the high value should be (yikes)
    # maybe add the scoring, but let's see what happens without
    search = GridSearchCV(
        kmeans,
        param_grid,
        cv=5
    )
    search.fit(imgFeatures)
    k = search.best_params_['n_clusters']
    print("best k (%d) for data size (%d)" % k, len(imgFeatures))
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(imgFeatures)

def compute_stats_on_clusters(labels_, data_frame):
    # [0, 0, 1, 1, 2, 0] <- labels_
    # ["/path/img1.png", "/path/img2.png", ... ]
    # so let's group (cluster) the paths
# score, price, minInstalls, genre
    df = pd.DataFrame({
        'Clusters' : labels_,
        'Data'    : data_frame.to_numpy()
    })

    cluster_groups = df.groupby(['Clusters'], as_index=False)['Data'].agg({'list':(lambda x : list(x))})


if __name__ == '__main__':
    data_frame = get_data_frame()
    data_features = get_img_features(data_frame)
    compute_stats_on_clusters(cluster_imgs(data_features), data_frame)
