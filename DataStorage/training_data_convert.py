import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

def create_lexicon():
	lexicon = []
	intents = []
	with open('./DataStorage/RawData/sampleall.csv', 'r', encoding="utf8") as train_file:
		tabs = csv.reader(train_file)
		for tab in tabs:
			all_words = word_tokenize(tab[0].lower())
			all_intents = word_tokenize(tab[1].lower())
			lexicon += list(all_words)
			intents += list(all_intents)
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	intents = [lemmatizer.lemmatize(i) for i in intents]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#Might need to tweek this if statement, was if 1000 > w_counts[w] > 50:
		if 1000 > w_counts[w]:
			l2.append(w)
	i_counts = Counter(intents)
	i2 = []
	for i in i_counts:
		i2.append(i)
	return(l2, i2)


def sample_handling(lexicon, intents):
	with open('./DataStorage/RawData/sampleall.csv', 'r') as train_file:
		tabs = csv.reader(train_file)
		tabs_features = []
		intents_array = []
		for tab in tabs:
			current_words = word_tokenize(tab[0].lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			intents_amount = np.zeros(len(intents))
			intent_index_value = intents.index(tab[1].lower())
			intents_amount[intent_index_value] += 1
			intents_array.append(intents_amount)

			len_row = len(current_words)
			length_of_row = 15
			if len_row < length_of_row:
				left_over = length_of_row - len_row

			feature_set = []
			for steps_in_words in range(length_of_row):
				features = np.zeros(len(lexicon))
				if steps_in_words < len(current_words):
					if current_words[steps_in_words].lower() in lexicon:
						index_value = lexicon.index(current_words[steps_in_words].lower())
						features[index_value] += 1
				feature_set.append(features)
			tabs_features.append(feature_set)

		return tabs_features, intents_array

def create_feature_sets_and_labels(test_size=0.1):
	lexicon, intents = create_lexicon()
	queries, intents_array = sample_handling(lexicon, intents)

	training_data = []
	training_data.append([])
	training_data.append([])

	for i in range(len(queries)):
		line = []
		for word in queries[i]:
			line.append(word)
		intent_move = intents_array[i]
		line = np.reshape(line, (len(line) * len(line[0])))
		training_data[0].append(line)
		training_data[1].append(intent_move)

	#random.shuffle(training_data)
	#Partitioning features array into traing and test arrays
	training_data = np.array(training_data)
	testing_size = int(test_size*len(training_data[0]))

	#print(features)
	train_x = list(training_data[0][:-testing_size])
	train_y = list(training_data[1][:-testing_size])

	test_x = list(training_data[0][-testing_size:])
	test_y = list(training_data[1][-testing_size:])

	return train_x, train_y, test_x, test_y

def main():
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels()
    with open('./DataStorage/DataForML/firefly_train.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
