import os
import re
import math
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from extract_fetures import Tfidf_method, split_data, BOW_method, MY_method

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

def plot_hyperparameter_accuracy(hyperparameters, accuracies):
    plt.plot(hyperparameters, accuracies)
    plt.xlabel('Hyperparameter (Laplace Smoothing)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Hyperparameter')
    plt.show()

def run_naive_bayes(laplace_smoothing):
    tfidf = Tfidf_method()
    X, y, data = tfidf.preprocess_data()
    X, y = tfidf.extract_top_k_features(X, y, 1500, data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(y_train))
    naive_bayes.train(X_train, y_train)
    naive_bayes_predictions = naive_bayes.test(X_test)
    naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
    return naive_bayes_accuracy

def run_naive_bayes1(laplace_smoothing):
    bow = BOW_method()
    X, y, data = bow.preprocess_data()
    X, y = bow.extract_top_k_features(X, y, 150)
    X_train, X_test, y_train, y_test = split_data(X, y)
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(y_train))
    naive_bayes.train(X_train, y_train)
    naive_bayes_predictions = naive_bayes.test(X_test)
    naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
    return naive_bayes_accuracy

def run_naive_bayes2(laplace_smoothing):
    my = MY_method()
    X, y, data = my.preprocess_data()
    X, y = my.extract_top_k_features(X, y, 1500)
    X_train, X_test, y_train, y_test = split_data(X, y)
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(y_train))
    naive_bayes.train(X_train, y_train)
    naive_bayes_predictions = naive_bayes.test(X_test)
    naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
    return naive_bayes_accuracy

laplace_smoothing_values = [0.1, 0.5, 1.0, 10, 100]
accuracies = []
accuracies1 = []
accuracies2 = []

for laplace_smoothing in laplace_smoothing_values:
    accuracy = run_naive_bayes(laplace_smoothing)
    accuracy1 = run_naive_bayes1(laplace_smoothing)
    accuracy2 = run_naive_bayes2(laplace_smoothing)
    
    accuracies.append(accuracy)
    accuracies1.append(accuracy1)
    accuracies2.append(accuracy2)
    
    print(f"TF-IDF Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy}%")
    print(f"BOW Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy1}%")
    print(f"MY Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy2}%")

plt.plot(laplace_smoothing_values, accuracies, label='TF-IDF')
plt.plot(laplace_smoothing_values, accuracies1, label='BOW')
plt.plot(laplace_smoothing_values, accuracies2, label='MY')
plt.xlabel('Hyperparameter (Laplace Smoothing)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Hyperparameter')
plt.legend()
plt.show()
