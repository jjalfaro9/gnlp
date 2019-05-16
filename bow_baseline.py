# bag of words baseline

# first need to set up vocabulary

# train and fit the model with those extractions

# then use the test data to predict what images should be given

# you have a gold standard of the image
# so you can save them and manually verify what happened
import numpy as np
import argparse
import json 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from utility import *

parser = argparse.ArgumentParser()
parser.add_argument("PNG_DIR", help="directory for all images")
parser.add_argument("metadata_path", help="file to all data")
args = parser.parse_args()

vectorizer = CountVectorizer(ngram_range=(1,2))

def get_trained_data_features(training_data):
    return vectorizer.fit_transform(training_data)

def get_test_data_features(test_data):
    return vectorizer.transform(test_data)

def get_full_corpus(metadata_path, PNG_DIR):
    return get_data_frame(metadata_path, PNG_DIR)

def split_corpus_test_train(data):
    descriptions = data['description'].to_numpy()
    icons = data['icon'].to_numpy()
    print("Len of descriptions (%s) and icons (%s) should be the same!" % (len(descriptions), len(icons)))
    print("Example of icon (icon livin' lol) %s " % icons[0])
    return train_test_split(descriptions, icons, train_size=0.6)


def do_some_stuff(meta, pngs):
    print("Getting all the data...")
    data = get_full_corpus(meta, pngs)
    print("Splitting into training and testing sets...")
    d_train, d_test, i_train, i_test = split_corpus_test_train(data)
    print("len of d_train and i_train should be samee %s %s" % (len(d_train), len(i_train)))
    print("Get bag of word features for training...")
    bow_train_features = get_trained_data_features(d_train)
    print("Get bag of word features for testing...")
    bow_test_features = get_test_data_features(d_test)

    neighbors = NearestNeighbors(metric='cosine')
    print("Fitting for NearestNeighbors...")
    neighbors.fit(bow_train_features)
    print("Finding NearestNeighbors for testing set...")
    results = neighbors.kneighbors(bow_test_features, return_distance=False)
    # results should be an array containing arrays (of max size five) which are index back into bow_train_features
    # results = [[a,b,c,d,e], [t], [q], [r,s], [j,m]]
    # first index corresponds to first test data image
    # test image features = [[], [], [], []]

    # features are in the same space, computed with the same vgg16 model
    print("Let's get some img features!")
    training_icon_f, testing_icon_f = get_features(pngs, i_train, i_test)
    img_n = NearestNeighbors(metric='cosine')
    # we need to only look up unseen img data!
    img_n.fit(testing_icon_f)
    print("len of results (%s) should be the same as len of testing_features (%s)" % (len(results), len(testing_icon_f)))
    # now that we have our k-nearest neighbors from bag of words features, look up the image features
    # now that we have those image features, we need to find the k-closest images to those images
    # accuracy is calculated by seeing if the k-closest images contain my golden image!
    accuracy = 0
    counter = 0
    img_map = dict()
    for i, r in enumerate(results):
        neighbors_f = [training_icon_f[j] for j in r]
        counter += 1

        img_neighbors = img_n.kneighbors(neighbors_f, return_distance=False)
        # okay, okay! so r has a len based on how many k neighbors we want, so for each of those we're finding j neighbors of the images
        # so img_neighbors is a k x j matrix!
        # okay! this contains the nearest neighbors [[], [], [], [], []] and within that are indecies
        d = {}
        closests_featured_imgs = [[testing_icon_f[n] for n in neighbor]for neighbor in img_neighbors]
        d['close_imgs'] = [ i_test[close_img] for neighbor in img_neighbors for close_img in neighbor ]
        if np.array(testing_icon_f[i]) in np.array(closests_featured_imgs):
            accuracy += 1
            d['golden_present'] = True
        else:
            d['golden_present'] = False
        if (counter % 5000 == 0):
            print("Counter: %s" % counter)
            print("Current accuracy: %s" % (accuracy/len(testing_icon_f)))
        img_map[i_test[i]] = d
    testing_len = len(testing_icon_f)
    train_len = len(training_icon_f)
    accuracy = accuracy / testing_len
    print("Accuracy for finding an image %f" % accuracy)
    with open('results.txt', 'w') as f:
        f.write(f'Accuracy for finding an image with bag of words baseline model: => {accuracy}\n')
        f.write(f'size of testing data: => {testing_len}, size of training data: => {train_len}')
    with open('results.json', 'w') as r:
        json.dump(img_map, r)



if __name__ == '__main__':
    do_some_stuff(args.metadata_path, args.PNG_DIR)
