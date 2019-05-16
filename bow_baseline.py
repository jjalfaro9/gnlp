# bag of words baseline

# first need to set up vocabulary

# train and fit the model with those extractions

# then use the test data to predict what images should be given

# you have a gold standard of the image
# so you can save them and manually verify what happened
import numpy as np
import argparse

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
    # train_i, test_i = train_test_split(np.arange(len(data)), train_size=0.6, random_state=44)
    # train = data.ix[train_i]
    # test = data.ix[test_i]
    # return (train, test)
    # d = data['description'].to_numpy() if data frame doesnt work
    # actually I don't want to lose the img png directory 'icon'
    # so if my assumption is correct, we can just go ahead and do this
    # why did this give an error?
    descriptions = data['description'].to_numpy()
    icons = data['icon'].to_numpy()
    return train_test_split(descriptions, icons, train_size=0.6)


def do_some_stuff(meta, pngs):
    print("Getting all the data...")
    data = get_full_corpus(meta, pngs)
    print("Splitting into training and testing sets...")
    # don't know if you can just do this. as data is a panda frame with array of dictionaries
    d_train, d_test, i_train, i_test = split_corpus_test_train(data)
    print("len of d_train and i_train should be samee %s %s" % (len(d_train), len(i_train)))
    print("Get bag of word features for training...")
    bow_train_features = get_trained_data_features(d_train)
    # getnnz()
    # print("Length of bow_train_features %s" % bow_train_features.getnnz())
    print("Get bag of word features for testing...")
    bow_test_features = get_test_data_features(d_test)
    # print("Length of bow_test_features %s" % bow_test_features.getnnz())

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
    training_icon_f, testing_icon_f = get_features(i_train, i_test)
    img_n = NearestNeighbors(metric='cosine')
    # put all the image features (this way we search the whole space when looking for close images)
    img_n.fit(training_icon_f + testing_icon_f)
    print("len of results (%s) should be the same as len of testing_features (%s)" % (len(results), len(testing_icon_f)))
    # now that we have our k-nearest neighbors from bag of words features, look up the image features
    # now that we have those image features, we need to find the k-closest images to those images
    # accuracy is calculated by seeing if the k-closest images contain my golden image!
    accuracy = 0
    counter = 0
    for i, r in enumerate(results):
        neighbors_f = map(lambda j: training_icon_f[j], r)
        counter += 1


        # do we fit all the values (?)
        img_neighbors = img_n.kneighbors(neighbors_f, return_distance=False)
        # now that we have the closest is the golden value in this (?)
        # we need to check and see the dimension of these things though, what's going on here
        if testing_icon_f[i] in img_neighbors:
            accuracy += 1
        if (counter % 5000 == 0):
            print("Counter: %s" % counter)
            print("Len & Shape of neighbors_f: (%s, %s) " % (len(neighbors_f),"idk"))
            print("Len & Shape of img_neighbors: (%s, %s)" % (len(img_neighbors), "idk"))
            print("Current accuracy: %s" % (accuracy/len(testing_icon_f)))
    accuracy = accuracy / len(testing_icon_f)
    print("Accuracy for finding an image %f" % accuracy)



if __name__ == '__main__':
    do_some_stuff(args.metadata_path, args.PNG_DIR)
