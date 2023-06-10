from logistic_regression import LogisticRegression
import math
import random
import matplotlib.pyplot as plt

from minist.extract_features import MNISTFeatures

features = MNISTFeatures()
X_train, Y_train, X_test, Y_test  = features.load_mnist_data()

pixel_X_train = [features.pixel_intensity_feature(row) for row in X_train]
pixel_X_test = [features.pixel_intensity_feature(row) for row in X_test]

hog_X_train = [features.hog_feature(row) for row in X_train]
hog_X_test = [features.hog_feature(row) for row in X_test]

pca = features.pca_feature(pixel_X_train)



pca_X_train = [features.extract_features(row, pca) for row in pixel_X_train]
pca_X_test = [features.extract_features(row, pca) for row in pixel_X_test]

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]

# Pixel features
pixel_accuracies = []
for learning_rate in learning_rates:
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(pixel_X_train, Y_train)
    accuracy = log_reg.predict(pixel_X_test, Y_test)
    pixel_accuracies.append(accuracy)

# HOG features 
hog_accuracies = []
for learning_rate in learning_rates:
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(hog_X_train, Y_train)
    accuracy = log_reg.predict(hog_X_test, Y_test)
    hog_accuracies.append(accuracy)
    
# PCA features
pca_accuracies = []
for learning_rate in learning_rates:
    log_reg = LogisticRegression(reg_lambda=0.01, iterations=30, alpha=learning_rate, num_classes=10)
    log_reg.fit(pca_X_train, Y_train)
    accuracy = log_reg.predict(pca_X_test, Y_test)
    pca_accuracies.append(accuracy)


plt.plot(learning_rates, pixel_accuracies)
plt.title('Pixel Accuracy vs. Learning Rate')


plt.plot(learning_rates, hog_accuracies)



plt.plot(learning_rates, pca_accuracies)


plt.show()