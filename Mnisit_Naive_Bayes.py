import os
import re
import math
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from NaiveBayesMain import NaiveBayes, accuracy_score

from minist.extract_features import MNISTFeatures
# from bbc.extract_fetures import Tfidf_method, split_data, BOW_method, MY_method

def plot_hyperparameter_accuracy(hyperparameters, accuracies):
    plt.plot(hyperparameters, accuracies)
    plt.xlabel('Hyperparameter (Laplace Smoothing)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Hyperparameter')
    plt.show()
    
features = MNISTFeatures()

X_train, Y_train, X_test, Y_test  = features.load_mnist_data()

pixel_X_train = [features.pixel_intensity_feature(row) for row in X_train]
pixel_X_test = [features.pixel_intensity_feature(row) for row in X_test]

hog_X_train = [features.hog_feature(row) for row in X_train]
hog_X_test = [features.hog_feature(row) for row in X_test]

pca = features.pca_feature(pixel_X_train)



pca_X_train = [features.extract_features(row, pca) for row in pixel_X_train]
pca_X_test = [features.extract_features(row, pca) for row in pixel_X_test]
def run_naive_bayesMnist1(laplace_smoothing):
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(Y_train))
    naive_bayes.train(pixel_X_train, Y_train)
    naive_bayes_predictions = naive_bayes.test(pixel_X_test)
    naive_bayes_accuracy = accuracy_score(Y_test, naive_bayes_predictions)
    return naive_bayes_accuracy


def run_naive_bayesMnist2(laplace_smoothing):
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(Y_train))
    naive_bayes.train(hog_X_train, Y_train)
    naive_bayes_predictions = naive_bayes.test(hog_X_test)
    naive_bayes_accuracy = accuracy_score(Y_test, naive_bayes_predictions)
    return naive_bayes_accuracy

def run_naive_bayesMnist3(laplace_smoothing):
    naive_bayes = NaiveBayes(laplace_smoothing=laplace_smoothing, total_data=len(Y_train))
    naive_bayes.train(pca_X_train, Y_train)
    naive_bayes_predictions = naive_bayes.test(pca_X_test)
    naive_bayes_accuracy = accuracy_score(Y_test, naive_bayes_predictions)
    return naive_bayes_accuracy

laplace_smoothing_values = [0.1, 0.5, 1.0, 10, 100]
accuracies = []
accuracies1 = []
accuracies2 = []

for laplace_smoothing in laplace_smoothing_values:
    accuracy = run_naive_bayesMnist1(laplace_smoothing)
    accuracy1 = run_naive_bayesMnist2(laplace_smoothing)
    accuracy2 = run_naive_bayesMnist3(laplace_smoothing)
    
    accuracies.append(accuracy)
    accuracies1.append(accuracy1)
    accuracies2.append(accuracy2)
    
    print(f"Pixel Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy}%")
    print(f"hog Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy1}%")
    print(f"pca Accuracy with Laplace Smoothing = {laplace_smoothing}: {accuracy2}%")

plt.plot(laplace_smoothing_values, accuracies, label='Pixel')
plt.plot(laplace_smoothing_values, accuracies1, label='hog')
plt.plot(laplace_smoothing_values, accuracies2, label='pca')
plt.xlabel('Hyperparameter (Laplace Smoothing)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Hyperparameter')
plt.legend()
plt.show()
