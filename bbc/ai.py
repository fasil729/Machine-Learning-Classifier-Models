import os
import re
import math
import random
from collections import Counter, defaultdict

def train_test_split(data, test_ratio=0.5):
    random.shuffle(data)
    test_size = int(len(data) * test_ratio)
    return data[test_size:], data[:test_size]


class NaiveBayes:
    def __init__(self, total_data, laplace_smoothing):
        self.laplace_smoothing = laplace_smoothing
        self.total_data = total_data
        self.memo = defaultdict(int)
        self.vocabulary = defaultdict(set)
        self.label_frequency = defaultdict(int) 

    def calculate_probability(self, frequency, num_events, voc_size):
        probability = (frequency + self.laplace_smoothing)/(num_events + self.laplace_smoothing*voc_size)
        probability = math.log(probability)
        return probability


    def train(self, feature_matrix, classes):
        
        for features, label in zip(feature_matrix, classes):
            self.label_frequency[label] += 1
        
            for feature_idx, feature_value in enumerate(features):
                memo_key = (feature_idx, feature_value, label)
                self.vocabulary[feature_idx].add(feature_value)
                self.memo[memo_key] += 1

    
    def test(self, feature_matrix):
        predictions = []
        num_unique_classes = len(self.label_frequency)

        for features in feature_matrix:
            probabilities = []
        
            for label, frequency in self.label_frequency.items():
                class_probability = self.calculate_probability(frequency, self.total_data, num_unique_classes)

                probability = class_probability
                for feature_idx, feature_value in enumerate(features):
                    # p(feature/label)
                    feature = (feature_idx, feature_value, label)
                    feature_probability = self.calculate_probability(self.memo[feature], frequency, len(self.vocabulary[feature_idx]))
                    probability += feature_probability

                probabilities.append((label, probability))

            predictions.append(max(probabilities, key=lambda x: x[1])[0])

        return predictions




# Evaluation
def accuracy_score(y_true, y_pred):

    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return (correct / len(y_true))*100

 # Main function to demonstrate the classifiers

# doc_class = {}

# filename = 'bbc/bbc.classes'
# with open(filename, 'r') as file:
#     for ind, line in enumerate(file):        
#         if ind > 3:
#             doc, _class = map(int, line.split())
#             doc_class[doc + 1] = _class

# matrix = defaultdict(lambda: defaultdict(float))
# filename = 'bbc/bbc.mtx'

# unique_words = defaultdict(int)


# with open(filename, 'r') as file:
#     for ind, line in enumerate(file):
        
#         line = line.strip()
#         if ind > 1 and line:
#             word, doc, freq = line.split()
#             matrix[int(doc)][int(word)] = float(freq)
#             unique_words[word] += 1
            



# datasets = []
# for doc, words in matrix.items():
#     dataset = [0]*9636
#     for word, freq in words.items():
#         dataset[word - 1] = freq
    
#     dataset[-1] = doc_class[doc]
#     datasets.append(dataset)



# train, test = train_test_split(datasets)

# # X_train = [x[:-1] for x in train]
# # y_train = [x[-1] for x in train]
# # X_test = [x[:-1] for x in test]
# # y_test = [x[-1] for x in tes#t]

from extract_fetures import Tfidf_method, split_data, BOW_method, MY_method
tfidf = Tfidf_method()
X, y, data = tfidf.preprocess_data()
X, y = tfidf.extract_top_k_features(X, y, 1500, data)
X_train, X_test, y_train, y_test = split_data(X, y)
naive_bayes = NaiveBayes(laplace_smoothing=0.5, total_data=len(y_train))
naive_bayes.train(X_train, y_train)
naive_bayes_predictions = naive_bayes.test(X_test)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
print("using tfidf method of extraction")
print("Naive Bayes Accuracy:", naive_bayes_accuracy, '%')



bow = BOW_method()
X, y, data = bow.preprocess_data()
X, y = bow.extract_top_k_features(X, y, 150)
X_train, X_test, y_train, y_test = split_data(X, y)
naive_bayes = NaiveBayes(laplace_smoothing=0.5, total_data=len(y_train))
naive_bayes.train(X_train, y_train)
naive_bayes_predictions = naive_bayes.test(X_test)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
print("using bow method of extraction")
print("Naive Bayes Accuracy:", naive_bayes_accuracy, '%')




my = MY_method()
X, y, data = my.preprocess_data()
X, y = my.extract_top_k_features(X, y, 1500)
X_train, X_test, y_train, y_test = split_data(X, y)
naive_bayes = NaiveBayes(laplace_smoothing=0.5, total_data=len(y_train))
naive_bayes.train(X_train, y_train)
naive_bayes_predictions = naive_bayes.test(X_test)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
print("using bow method of extraction")
print("Naive Bayes Accuracy:", naive_bayes_accuracy, '%')













