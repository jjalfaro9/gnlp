import faiss
import pickle
import json

# first we need to index put the vectors in there
# then we can search things up!

# the indexed vectors need to be of size nb x d and query vectors need to be nq x d

def _get_index(dimension, elements):
    index = faiss.IndexFlatL2(dimension)
    print("Is index trained? (%s)" % index.is_trained)
    index.add(elements)
    print("Total number of data elements indexed = %s" % index.ntotal)
    return index

def _search_index(index, elements, k):
    return index.search(elements, k)

def _faiss_get_nearest_k_neighbors(dimension, indexElements, searchElements, k):
    index = _get_index(dimension, indexElements)
    # I - result of this operation can be conveniently stored in an integer matrix of size nq-by-k,
        # where row i contains the IDs of the neighbors of query vector i, sorted by increasing distance
    # D - nq-by-k floating-point matrix with the corresponding squared distances.
    D, I = _search_index(index, searchElements, k)
    # so what this is saying is that for the first search vector, the elements in here are IDs in the freaking database
    # plus what's nice is that they are already sorted lol, wow this is super easy
    return I

def _evaluate(I, gold):
    correct = 0
    results = []
    for index, neighbors in enumerate(I):
        if gold in neighbors:
            correct += 1
            results.append() # need to think about this a bit more !
    return correct, results

def _general_algo(txt_train_elements, txt_test_elements, img_training_fts, img_testing_fts, img_icons, k):
    bow_dimension = txt_train_elements.shape[1]
    img_dimension = img_testing_fts.shape[1]

    # bow_I (nq x k) index 0 is index 0 of txt_test_elements and this row contains indecies to txt_train_elements
    # 10k x (100, 1000)
    bow_I = _faiss_get_nearest_k_neighbors(bow_dimension, txt_train_elements, txt_test_elements, k)


    accuracy = 0
    len_of_img_I = 0
    results_metadata = dict()
    # loop iterates for 10k times
    for test_idx, row in enumerate(bow_I):
        bow_img_neighbors = [img_training_fts[i] for i in row]
        # img_I now has (nq x k) index 0 is 0 of bow_img_neighbors and this row contains ids in img_testing_fts
        # (100, 1000) x (100, 1000)
        # 1000,000 or 10,000,000
        img_I = _faiss_get_nearest_k_neighbors(img_dimension, img_testing_fts, bow_img_neighbors, k)
        len_of_img_I += len(img_I)
        results = _evaluate(img_I, test_idx)
        accuracy += results[0]
        d = {}
        d['neighbors'] = {img_icons[neighbor] for neighbors in img_I for neighbor in neighbors}
        if results[0] != 0:
            d['golden_present'] = True
        else:
            d['golden_present'] = False
        results_metadata[img_icons[test_idx]] = d
    return (accuracy, results_metadata)

def _read_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
def _write_metrics(filename, k, accuracy, metadata, training_len, testing_len):
    with open(f'{filename}-{k}.json', 'w') as d:
        json.dump(metadata, d)
    with open(f'{filename}-{k}.txt', 'w') as f:
        f.write(f'Accuracy is {accuracy} for top k {k} \n')
        f.write(f'training set of {training_len} and testing set of {testing_len}\n')
        overall_acc = accuracy / len(testing_len)
        f.write(f'over all accuracy {overall_acc}')

def _compute_accuracies_metadata(name, description_fts_train, description_fts_test, img_fts_train, img_fts_test, img_icon_train, img_icon_test):
    print("All of these should be 100k (%d)(%d)(%d)" %(description_fts_train, img_fts_train, img_icon_train))
    print("All of these should be 10k (%d)(%d)(%d)" % (description_fts_test, img_fts_test, img_icon_test))
    # need to just figure out what to do with the results from general algo
    acc_100, meta_100 =  _general_algo(description_fts_train, description_fts_test, img_fts_train, img_fts_test, img_icon_test, 100) # Top 1%
    acc1000, meta_1000 = _general_algo(description_fts_train, description_fts_test, img_fts_train, img_fts_test, img_icon_test, 1000) # Top 10%

    _write_metrics(name, 100, acc_100, meta_100, len(img_fts_train), len(img_fts_test))
    _write_metrics(name, 1000, acc1000, meta_1000, len(img_fts_train), len(img_fts_test))


def bow():
    # these files are hardcoded and should be on the system

    # words, and png directories
    description_train = np.array(_read_file('../bow/description_train.p')[0:100000])
    description_test  = np.array(_read_file('../bow/description_test.p')[0:10000])
    img_icon_train = np.array(_read_file('../bow/icon_train.p')[0:100000])
    img_icon_test = np.array(_read_file('../bow/icon_test.p')[0:10000])

    # features
    bow_fts_train = np.array(_read_file('../bow/bow_fts_train.p')[0:100000])
    bow_fts_test = np.array(_read_file('../bow/bow_fts_test.p')[0:10000])
    img_fts_train = np.array(_read_file('../bow/img_fts_train.p')[0:100000])
    img_fts_test = np.array(_read_file('../bow/img_fts_test.p')[0:10000])

    _compute_accuracies_metadata("bow", bow_fts_train, bow_fts_test, img_fts_train, img_fts_test, img_icon_train, img_icon_test)

def bert():
    # features
    bert_vec_train = np.array(_read_file('../bert_fts/bert_vec_train.p')[0:100000])
    bert_vec_test = np.array(_read_file('../bert_fts/bert_vec_test.p')[0:10000])
    img_fts_train = np.array(_read_file('../bert_fts/img_fts_train.p')[0:100000])
    img_fts_test = np.array(_read_file('../bert_fts/img_fts_test.p')[0:10000])

    #png directories
    img_icon_train = np.array(_read_file('../bert_fts/img_icons_train.p')[0:100000])
    img_icon_test  = np.array(_read_file('../bert_fts/img_icons_test.p')[0:10000])

    _compute_accuracies_metadata("bert", bert_vec_train, bert_vec_test, img_fts_train, img_fts_test, img_icon_train, img_icon_test)

if __name__ == '__main__':
    bow()
    bert()
