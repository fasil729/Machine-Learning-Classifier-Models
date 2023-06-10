import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from Weather.weather_features import (
    extract_features_basic,
    extract_features_numeric,
    extract_features_categorical,
    x_train_basic,x_test_basic,x_train_numeric,x_train_categorical,x_test_numeric,x_test_categorical,
    read_data_from_csv,
    x_train,x_test
    
)
from NaiveBayesMain import NaiveBayes, accuracy_score, train_test_split

def run_naive_bayesWeather1(laplace_smoothing):
    # Train and test the Naive Bayes classifier using basic features
    nb = NaiveBayes(len(train_data), laplace_smoothing)
    nb.train(x_train_basic, [row[-1] for row in train_data])
    predictions = nb.test(x_test_basic)
    accuracy = accuracy_score([row[-1] for row in test_data], predictions)
    return accuracy

def run_naive_bayesWeather2(laplace_smoothing):
    # Train and test the Naive Bayes classifier using numeric features
    nb = NaiveBayes(len(train_data), laplace_smoothing)
    nb.train(x_train_numeric, [row[-1] for row in train_data])
    predictions = nb.test(x_test_numeric)
    accuracy = accuracy_score([row[-1] for row in test_data], predictions)
    return accuracy

def run_naive_bayesWeather3(laplace_smoothing):
    # Train and test the Naive Bayes classifier using categorical features
    nb = NaiveBayes(len(train_data), laplace_smoothing)
    nb.train(x_train_categorical, [row[-1] for row in train_data])
    predictions = nb.test(x_test_categorical)
    accuracy = accuracy_score([row[-1] for row in test_data], predictions)
    return accuracy

# Example usage
file_path = "data/weather.csv"

# Read the data from the CSV file
data = read_data_from_csv(file_path)

# Split the data into train and test sets
train_data = x_train
test_data = x_test

# Feature extraction
feature_matrix_basic = extract_features_basic(train_data)
feature_matrix_numeric = extract_features_numeric(train_data)
feature_matrix_categorical = extract_features_categorical(train_data)

# Hyperparameter tuning
laplace_smoothing_values = [0.1, 0.5, 1.0, 10, 100]
accuracies = []
accuracies1 = []
accuracies2 = []

for laplace_smoothing in laplace_smoothing_values:
    accuracy = run_naive_bayesWeather1(laplace_smoothing)
    accuracy1 = run_naive_bayesWeather2(laplace_smoothing)
    accuracy2 = run_naive_bayesWeather3(laplace_smoothing)
    
    accuracies.append(accuracy)
    accuracies1.append(accuracy1)
    accuracies2.append(accuracy2)
    
    print(f"Accuracy (Basic Features) with Laplace Smoothing = {laplace_smoothing}: {accuracy}%")
    print(f"Accuracy (Numeric Features) with Laplace Smoothing = {laplace_smoothing}: {accuracy1}%")
    print(f"Accuracy (Categorical Features) with Laplace Smoothing = {laplace_smoothing}: {accuracy2}%")

plt.plot(laplace_smoothing_values, accuracies, label='Basic Features')
plt.plot(laplace_smoothing_values, accuracies1, label='Numeric Features')
plt.plot(laplace_smoothing_values, accuracies2, label='Categorical Features')
plt.xlabel('Hyperparameter (Laplace Smoothing)')
plt.ylabel('Accuracy')
plt.title('Weather demo')
plt.legend()
plt.show()
