from logistic_regression import LogisticRegression
import math
import random
import matplotlib.pyplot as plt

from bbc.extract_fetures import Tfidf_method, split_data, BOW_method, MY_method

def run_for_bbc():
    features = BOW_method()
    X, y, data = features.preprocess_data()
    X, y = features.extract_top_k_features(X, y, 150)
    bow_X_train, bow_X_test, bow_y_train, bow_y_test = split_data(X, y)
    features = Tfidf_method()
    X1, y1, data = features.preprocess_data()
    X1, y1 = features.extract_top_k_features(X1, y1, 150, data)
    tfidf_X1_train, tfidf_X1_test, tfidf_y1_train, tfidf_y1_test = split_data(X1, y1)
    features = MY_method()
    X2, y2, data = features.preprocess_data()
    X2, y2 = features.extract_top_k_features(X2, y2, 150)
    my_X2_train, my_X2_test, my_y2_train, my_y2_test = split_data(X2, y2)

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
    
    # Pixel features
    bow_accuracies = []
    for learning_rate in learning_rates:
        log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
        log_reg.fit(bow_X_train, bow_y_train)
        accuracy = log_reg.predict(bow_X_test, bow_y_test)
        bow_accuracies.append(accuracy)
    
    # HOG features 
    tfidf_accuracies = []
    for learning_rate in learning_rates:
        log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
        log_reg.fit(tfidf_X1_train, tfidf_y1_train)
        accuracy = log_reg.predict(tfidf_X1_test, tfidf_y1_test)
        tfidf_accuracies.append(accuracy)
        
    # PCA features
    my_accuracies = []
    for learning_rate in learning_rates:
        log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
        log_reg.fit(my_X2_train, my_y2_train)
        accuracy = log_reg.predict(my_X2_test, my_y2_test)
        my_accuracies.append(accuracy)

    
    plt.plot(learning_rates, tfidf_accuracies, label='TF-IDF')
    plt.plot(learning_rates, bow_accuracies, label='BOw')
    plt.plot(learning_rates, my_accuracies, label='OUR Method')
    plt.xlabel('Hyperparameter (Learning Rate)')
    plt.ylabel('Accuracy')
    plt.title('BBC Text Classification(Logistic Regression)')
    plt.legend()
    plt.show()


run_for_bbc()


# learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
# accuracies = [21.80, 70.79, 90.56, 93.48, 93.48, 93.48]
# accuracies1 = [13.93, 22.47, 44.49, 49.44, 54.16, 55.28]
# accuracies2 = [12.36, 21.35, 31.91, 40.90, 44.27, 45.39]

# plt.plot(learning_rates, accuracies, label='BOW')
# plt.plot(learning_rates, accuracies1, label='TF-IDF')
# plt.plot(learning_rates, accuracies2, label='OUR Method')
# plt.xlabel('Hyperparameter (Learning Rate)')
# plt.ylabel('Accuracy')
# plt.title('BBC Text Classification(Logistic Regression)')
# plt.legend()
# plt.show()

