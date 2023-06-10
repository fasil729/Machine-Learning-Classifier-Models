import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from Weather.weather_features import (
    extract_features_basic,
    extract_features_numeric,
    extract_features_categorical,
    read_data_from_csv,
    train_data,
    test_data
)
from logistic_regression import LogisticRegression


def train_test_split(data, test_ratio=0.5):
    random.shuffle(data)
    test_size = int(len(data) * test_ratio)
    return data[test_size:], data[:test_size]


def run_logistic_weather1(learning_rate):
    # Train and test the Logistic Regression classifier using basic features
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(x_train_basic, [row[-1] for row in train_data])
    accuracy = log_reg.predict(x_test_basic, [row[-1] for row in test_data])
    return accuracy


def run_logistic_weather2(learning_rate):
    # Train and test the Logistic Regression classifier using numeric features
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(x_train_numeric, [row[-1] for row in train_data])
    accuracy = log_reg.predict(x_test_numeric, [row[-1] for row in test_data])
    return accuracy


def run_logistic_weather3(learning_rate):
    # Train and test the Logistic Regression classifier using categorical features
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(x_train_categorical, [row[-1] for row in train_data])
    accuracy = log_reg.predict(x_test_categorical, [row[-1] for row in test_data])
    return accuracy


# Example usage
file_path = "Weather/weather.csv"

# Read the data from the CSV file
data = read_data_from_csv(file_path)

# Split the data into train and test sets
train_data, test_data = train_test_split(data)

# Feature extraction
x_train_basic = extract_features_basic(train_data)
x_train_numeric = extract_features_numeric(train_data)
x_train_categorical = extract_features_categorical(train_data)

x_test_basic = extract_features_basic(test_data)
x_test_numeric = extract_features_numeric(test_data)
x_test_categorical = extract_features_categorical(test_data)

# Hyperparameter tuning
learning_rate_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracies = []
accuracies1 = []
accuracies2 = []

for learning_rate in learning_rate_values:
    accuracy = run_logistic_weather1(learning_rate)
    accuracy1 = run_logistic_weather2(learning_rate)
    accuracy2 = run_logistic_weather3(learning_rate)

    accuracies.append(accuracy)
    accuracies1.append(accuracy1)
    accuracies2.append(accuracy2)

    print(f"Accuracy (Basic Features) with Laplace Smoothing = {learning_rate}: {accuracy}%")
    print(f"Accuracy (Numeric Features) with Laplace Smoothing = {learning_rate}: {accuracy1}%")
    print(f"Accuracy (Categorical Features) with Laplace Smoothing = {learning_rate}: {accuracy2}%")

plt.plot(learning_rate_values, accuracies, label='Basic Features')
plt.plot(learning_rate_values, accuracies1, label='Numeric Features')
plt.plot(learning_rate_values, accuracies2, label='Categorical Features')
plt.xlabel('Hyperparameter (Laplace Smoothing)')
plt.ylabel('Accuracy')
plt.title('Weather Demo (Logistic Regression)')
plt.legend()
plt.show()
