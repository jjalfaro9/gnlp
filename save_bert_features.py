import pandas as pd
import numpy as np
import pickle

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split


def get_vectors(datas_index, sub_index, datas):
    #     print(app_index)
    #     (datas_index, sub_index) = metadata[app_index]['lookup_index']
    result = datas[datas_index][sub_index]
    x = np.zeros(shape)
    for j in range(len(result[:max_len])):
        x[j] = result[j]
    return x

def _get_data(image_directory):
    metadata = pickle.load(open("../bert/data_5k_bert.pkl", 'rb'))
    datas = []
    for i in range(23):
        print(i)
        datas.append(pickle.load(open(f"../bert/data_5k_bert_chunk_{i}.pkl", 'rb')))
    lookup_dict = []
    last_index = 0
    data_index = 0

    max_len = 30
    lstm_size = 200
    embedding_dim = len(datas[0][0][0])
    shape = (max_len, embedding_dim)

    pddata = pd.DataFrame(metadata)
    pddata['index_2'] = range(len(pddata))

    for i in range(len(metadata)):
        #     print(i)
        index = metadata[i]['index']
        index_2 = i
        if last_index > index:
            # new datapiece
            data_index += 1
        metadata[i]['lookup_index'] = (data_index, index)
        metadata[i]['y_value_index'] = i
        #     lookup_dict.append()
        last_index = index
    lookup_vectors = lookup_vectors = [x['lookup_index'] for x in metadata]
    pos_wordvectors = [get_vectors(x[0], x[1], datas) for x in lookup_vectors]
    image_paths = [x['icon'] for x in metadatas]
    image_objects = [image.load_img(f'{image_directory}/{img_path}', target_size=(224, 224)) for img_path in image_paths]
    x_images = [image.img_to_array(x) for x in image_objects]
    x_images_exp = [np.expand_dims(img_data, axis=0) for img_dat in x_images]
    imgs = [preprocess_input(img_d) for img_d in x_images_exp]
    vgg = VGG16(weights='imagenet', include_top=False)
    img_fts = vgg.predict(imgs)
    flattend_img_fts = [np.array(vg_ft).flatten() for vg_ft in img_fts]
    return (pos_wordvectors, flattend_img_fts, image_paths)

def _split_corpus_test_train(bert_vecs, img_fts, img_paths):

    descriptions_len, icons_len = len(bert_vecs), len(img_fts)
    if descritpions_len == 112282:
        ts = 0.9
    else:
        # neeed to compute ts => x / len = ts , 110k / len = (?) but what if len is not long enough
        if len > 100000:
            ts = 100000 / descritpions_len
        else:
            ts = 0.8
    return train_test_split(bert_vecs, img_fts, train_size=ts)

def _save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def save_features(image_directory):
    bert_vec, img_fts, img_paths = _get_data(image_directory)
    bert_vec_train, bert_vec_test, img_fts_train, img_fts_test, img_icons_train, img_icons_test = _split_corpus_test_train(bert_vec, img_fts, img_paths)

    _save_file('bert_vec_train.p', bert_vec_train)
    _save_file('bert_vec_test.p', bert_vec_test)
    _save_file('img_fts_train.p', img_fts_train)
    _save_file('img_fts_test.p', img_fts_test)
    _save_file('img_icons_train.p', img_icons_train)
    _save_file('img_icons_test.p', img_icons_test)
