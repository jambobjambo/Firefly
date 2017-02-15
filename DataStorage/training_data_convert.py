import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

def create_lexicon():
    lexicon = []
    with open('./DataStorage/RawData/training_data.tsv', 'r', encoding="utf8") as train_file:
        training = [line.strip().split('\t') for line in train_file]
        for row in training:
            all_words = word_tokenize(row[0].lower())
            lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        #Might need to tweek this if statement, was if 1000 > w_counts[w] > 50:
        if 1000 > w_counts[w]:
            l2.append(w)

    return(l2)


def sample_handling(lexicon):
    featureset = []
    with open('./DataStorage/RawData/training_data.tsv', 'r') as train_file:
        training = [line.strip().split('\t') for line in train_file]
        for row in training:
            current_words = word_tokenize(row[0].lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            classification = []
            if row[1] == "Revenue_Query":
                classification =[0,0,1]
            elif row[1] == "Summery_Query":
                classification =[0,1,0]
            elif row[1] == "Profit_Query":
                classification =[1,0,0]
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(test_size=0.1):
    lexicon = create_lexicon()
    features = []
    features += sample_handling(lexicon)
    random.shuffle(features)

    #Partitioning features array into traing and test arrays
    features = np.array(features)
    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,0][:-testing_size])

    test_x = list(features[:,0][:-testing_size])
    test_y = list(features[:,0][:-testing_size])

    return train_x, train_y, test_x, test_y

def main():
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels()
    with open('./DataStorage/DataForML/firefly_train.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
