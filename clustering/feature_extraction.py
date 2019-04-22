from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

import argparse
import os
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("imgsDir", help="directory for all images")
args = parser.parse_args()

# the only problem with this is that there is a lot of data, so idk if this will blow up
def get_img_features(imgDir):
    model = VGG16(weights='imagenet')

    dir = os.fsencode(imgDir)
    vgg16_features = []
    pathToFeatures = []

    for file in os.listdir(dir):
        fileName = os.fsdecode(file)
        if fileName.endswith(".png"):
            img = image.load_img(fileName, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            vgg16_feature = model.predict(img_data)
            vgg16_features.append(np.array(vgg16_feature).flatten())
            pathToFeatures.append(os.path.abspath(fileName))

    return (np.array(vgg16_features), pathToFeatures)


def cluster_imgs(imgFeatures):
    kmeans = KMeans()
    param_grid = {"n_clusters": range(2, 110)}
    # maybe add the scoring, but let's see what happens without
    search = GridSearchCV(
        kmeans,
        param_grid
    )
    search.fit(imgFeatures)
    return kmeans.labels_

def group_imgs(labels_, pathToFeatures):
    # [0, 0, 1, 1, 2, 0] <- labels_
    # ["/path/img1.png", "/path/img2.png", ... ]
    # so let's group (cluster) the paths

    df = pd.DataFrame({
        'Clusters' : labels_,
        'paths'    : pathToFeatures
    })
    i = 0
    for cluster in df.groupby(['Clusters'], as_index=False)['Paths'].agg({'list':(lambda x : list(x))}).list:
        newDir = os.getcwd() + "/cluster-" + str(i)
        i += 1
        if os.path.exists(newDir):
            shutil.rmtree(newDir)
        os.makedirs(newDir)
        for f in cluster:
            shutil.copy(f, newDir)


if __name__ == '__main__':
    f, p = get_img_features(args.imgsDir)
    group_imgs(cluster_imgs(f), p)
