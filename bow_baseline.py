# bag of words baseline

# first need to set up vocabulary

# train and fit the model with those extractions

# then use the test data to predict what images should be given

# you have a gold standard of the image
# so you can save them and manually verify what happened
import numpy as np
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
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
    descriptions = data['description'].to_numpy()
    icons = data['icon'].to_numpy()
    return train_test_split(descriptions, icons, train_size=0.6, random_sate=44)


def do_some_stuff():
    print("Getting all the data...")
    data = get_full_corpus("", "")
    print("Splitting into training and testing sets...")
    # don't know if you can just do this. as data is a panda frame with array of dictionaries
    d_train, d_test, i_train, i_test = split_corpus_test_train(data)
    print("len of d_train and i_train should be samee %s %s" % len(d_train), len(i_train))
    print("Get bag of word features for training...")
    bow_train_features = get_trained_data_features(d_train)
    print("Length of bow_train_features %s" % len(bow_train_features))
    print("Get bag of word features for testing...")
    bow_test_features = get_test_data_features(d_test)
    print("Length of bow_test_features %s" % len(bow_test_features))

    neighbors = NearestNeighbors(metric='cosine')
    print("Fitting for NearestNeighbors...")
    neighbors.fit(bow_train_features)
    print("Finding NearestNeighbors for testing set...")
    results = neighbors.kneighbors(bow_test_features, return_distance=False)
    # results should be an array containing arrays (of max size five) which are index back into bow_train_features
    # results = [[a,b,c,d,e], [t], [q], [r,s], [j,m]]
    # first index corresponds to first test data image
    # test image features = [[], [], [], []]

    training_icon_f, testing_icon_f = get_features(i_train, i_test)
    print("len of results (%s) should be the same as len of testing_features (%s)" % len(results), len(testing_icon_f))
    highest = (float("-inf"), -1)
    lowest = (float("inf"), -1)
    avgs = []
    acc = 0
    for i, r in enumerate(results):
        # r should be a list of index values into bow_train_features, but can we just use those to get into i_train
        # get the i_test[i] features
        # get all i_train[r] features (r is array )
        # compute cosine similarity
        # get average
        # save these values
        # find highest avg & lowest and save those images somehow
        neighbors_f = map(lambda i: training_icon_f[i], r)
        # smaller the anlge, the higher the similarity
        cos_similarities = cos_cdist(testing_icon_f[i], neighbors_f)
        avg = np.average(cos_similarities)
        avgs.append(avg)
        if avg > highest[0]:
            highest[0] = avg
            highest[1] = i
        else if avg < lowest[0]:
            lowest[0] = avg
            lowest[1] = i
        if testing_icon_f[i] in neighbors_f:
            acc += 1

    print(avgs)
    print(acc)



if __name__ == '__main__':
    do_some_stuff(args.metadata_path, args.PNG_DIR)
