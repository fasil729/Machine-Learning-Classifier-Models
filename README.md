
# Project Overview

This project focuses on conducting experiments with different machine learning algorithms and feature extraction methods on various datasets. It includes three distinct projects:

1. MNIST Digit Recognition: The goal of this project is to develop a model that can accurately perceive digits 1-9 from a given image dataset. our projet  is  achieved a maximum accuracy of 94.5%.

1. BBC News Classification: In this project, the aim is to classify news articles from the BBC into different categories such as sport, politics, entertainment, business, and tech. The goal is to achieve a high accuracy rate. our project meets maximium 95% accuracy 

1. Weather Prediction Demo: This project involves predicting whether to play golf based on weather conditions. A model is trained on a weather dataset, and the objective is to accurately predict golf-playing decisions.

.

## Table of Contents

- [Naive Bayes Experiment on MNIST Dataset](#naive-bayes-experiment-on-mnist-dataset)
- [Logistic Regression Experiment on MNIST Dataset](#logistic-regression-experiment-on-mnist-dataset)
- [Logistic Regression Experiment on BBC Dataset](#logistic-regression-experiment-on-bbc-dataset)
- [Naive Bayes Experiment on BBC Dataset](#naive-bayes-experiment-on-bbc-dataset)
- [Naive Bayes Experiment on Demo Weather Dataset](#naive-bayes-experiment-on-demo-weather-dataset)

## Naive Bayes Experiment on MNIST Dataset

The Naive Bayes model was evaluated on the MNIST digit dataset using three feature extraction methods: Pixel Intensity, HOG, and PCA. The experiments revealed the following insights:

### Pixel Intensity Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the Pixel Intensity feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 75%      |
| 0.5               | 62%      |
| 1                 | 52%      |
| 1.5               | 47%      |

### HOG Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the HOG feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 74%      |
| 0.5               | 61%      |
| 1                 | 51%      |
| 1.5               | 46%      |

### PCA Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the PCA feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 14%      |
| 0.5               | 14%      |
| 1                 | 14%      |
| 1.5               | 13%      |

## Logistic Regression Experiment on MNIST Dataset

The Logistic Regression model was experimented with the same MNIST digit dataset using different feature extraction methods: Pixel Intensity, HOG, and PCA. The findings are as follows:

### Pixel Intensity Feature:

The table below summarizes the accuracy achieved by the Logistic Regression model with different Laplace smoothing values for the Pixel Intensity feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.0001            | 12%      |
| 0.001             | 12%      |
| 0.01              | 12%      |
| 0.1               | 73%      |
| 0.5               | 85%      |
| 1                 | 88%      |
| 1.5               | 84%      |

### HOG Feature:

The table below summarizes the accuracy achieved by the Logistic Regression model with different Laplace smoothing values for the HOG feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.0001            | 4.5%     |
| 0.001             | 4.5%     |
| 0.01              | 4.5%     |
| 0.1               | 74%      |
| 0.5               | 90%      |
| 1                 | 95%      |
| 1.5               | 94.5%    |

### PCA Feature:

The table below summarizes the accuracy achieved by the Logistic Regression model with different Laplace smoothing values for the PCA feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.0001            | 15.5%    |
| 0.001             | 15.5%    |
| 0.01              | 15.5%    |
| 0.1               | 76%      |
| 0.5               | 83%      |
| 1                 | 87%      |
| 1.5               | 87.5%    |

## Logistic Regression Experiment on BBC Dataset

The Logistic Regression model was evaluated on the BBC dataset using different feature extraction methods: Bag of Words, MY, andTF-IDF. Please refer to the project directory for the detailed experimental results and code.

## Naive Bayes Experiment on BBC Dataset

The Naive Bayes model was experimented with the BBC dataset using different feature extraction methods: Bag of Words, TF-IDF, and MY. The experiments revealed the following insights:

### Bag of Words (BoW) Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the Bag of Words feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 95%      |
| 0.5               | 91%      |
| 1                 | 88%      |
| 1.5               | 86%      |

### TF-IDF Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the TF-IDF feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 94%      |
| 0.5               | 90%      |
| 1                 | 87%      |
| 1.5               | 85%      |

### MY Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the MY feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 85%      |
| 0.5               | 83%      |
| 1                 | 80%      |
| 1.5               | 78%      |

## Naive Bayes Experiment on Demo Weather Dataset

The Naive Bayes model was evaluated on the Demo Weather dataset using different feature extraction methods: categorical, numerical, and basic. The experiments revealed the following insights:

### Categorical Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the categorical feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 85%      |
| 0.5               | 84%      |
| 1                 | 84%      |
| 1.5               | 83%      |

### Numerical Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the numerical feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 98%      |
| 0.5               | 82%      |
| 1                 | 66%      |
| 1.5               | 60%      |

### Basic Feature:

The table below summarizes the accuracy achieved by the Naive Bayes model with different Laplace smoothing values for the basic feature extraction method:

| Laplace Smoothing | Accuracy |
| ----------------- | -------- |
| 0.1               | 99%      |
| 0.5               | 87%      |
| 1                 | 86%      |
| 1.5               | 86.5%    |

Based on the insights gained from the experiments, it can be concluded that the choice of feature extraction method significantly impacts the performance of the Naive Bayes model on the BBC dataset and the Demo Weather dataset. For the BBC dataset, the Bag of Words feature extraction method achieved the highest accuracy of 95%, while for the Demo Weather dataset, the basic feature extraction method achieved the highest accuracy of 99%. Additionally, lower values of Laplace smoothing tend to yield higher accuracy across most feature extraction methods.

The Naive Bayes model achieved an accuracy of 92.3% for predicting whether to play golf based on the given weather conditions.
For more detailed information and analysis on each experiment, please refer to the [complete document here ↗](https://docs.google.com/document/d/1HMCUqvCfBRvUdsRB7WDFlVSpUBNG40FS1SbSK-BqMSQ/edit?usp=sharing).

Feel free to explore the document for a comprehensive understanding of the experiments and their results.
## Team

- Deribew Shimelis
- Enyew Anberbir
- Fasika Fikadu
- Kaleab Tigebu
- Rahel Solomon
