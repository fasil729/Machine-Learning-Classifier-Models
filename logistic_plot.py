import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mnist import MNIST
from skimage.feature import hog
from sklearn.decomposition import PCA

def load_mnist_data():
    mndata = MNIST('mnist_data')
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    return X_train[:600], y_train[:600], X_test[:200], y_test[:200]

def pixel_intensity_feature(image):
    return [x/255.0 for x in image]

def hog_feature(image):
    image = np.array(image).reshape(28, 28)
    feature_vector, _ = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    return feature_vector

def pca_feature(images, n_components=100):
    pca = PCA(n_components)
    pca.fit(images)
    return pca

def extract_features(image, pca):
    pixel_intensity = image
    pca_feat = pca.transform([image]).flatten()
    hog_feat = hog_feature(image)
    return hog_feat

X_train, Y_train, X_test, Y_test  = load_mnist_data()

X_train = [pixel_intensity_feature(row) for row in X_train]
X_test = [pixel_intensity_feature(row) for row in X_test]

pca1 = pca_feature(X_train)
pca2 = pca_feature(X_test)

X_train = [extract_features(row, pca1) for row in X_train]
X_test = [extract_features(row, pca2) for row in X_test]

print(len(X_train[0]))

def softmax(z):
    exp_z = [[math.exp(e) for e in row] for row in z]
    return [[e / sum(row) for e in row] for row in exp_z]

def compute_cost(X, y, W, b, reg_lambda):
    m, n = len(X), len(X[0])
    z = [[sum([X[i][k] * W[k][j] for k in range(n)]) + b[j] for j in range(len(b))] for i in range(m)]
    y_hat = softmax(z)
    y_one_hot = [[1 if c == y[i] else 0 for c in range(len(b))] for i in range(m)]
    total_cost = -(1/m) * sum([sum([y_one_hot[i][j] * math.log(y_hat[i][j]) for j in range(len(b))]) for i in range(m)])
    
    # Add L2 regularization term
    L2_reg = (reg_lambda / 2) * sum([sum([W[k][j]**2 for j in range(len(b))]) for k in range(n)])
    total_cost += L2_reg
    
    return total_cost

m, n = len(X_train), len(X_train[0])

def compute_gradient(X, y, W, b, reg_lambda):
    m, n = len(X), len(X[0])
    z = [[sum([X[i][k] * W[k][j] for k in range(n)]) + b[j] for j in range(len(b))] for i in range(m)]
    y_hat = softmax(z)
    y_one_hot = [[1 if c == y[i] else 0 for c in range(len(b))] for i in range(m)]
    
    dj_db = [(1/m) * sum([y_hat[i][j] - y_one_hot[i][j] for i in range(m)]) for j in range(len(b))]
    dj_dW = [[(1/m) * sum([(y_hat[i][j] - y_one_hot[i][j]) * X[i][k] for i in range(m)]) for j in range(len(b))] for k in range(n)]

    # Add L2 regularization term for gradients
    dj_dW = [[dj_dW[k][j] + reg_lambda * W[k][j] for j in range(len(b))] for k in range(n)]
    return dj_db, dj_dW

def gradient_descent(Xtest, Ytest, X, y, W_in, b_in, cost_function, gradient_function, alpha, num_iters, prediction_function, reg_lambda): 
    m = len(X)
    for i in range( num_iters):
        dj_db, dj_dW = gradient_function(X, y, W_in, b_in, reg_lambda)   

        W_in = [[W_in[k][j] - alpha * dj_dW[k][j] for j in range(len(b_in))] for k in range(n)]
        b_in = [b_in[j] - alpha * dj_db[j] for j in range(len(b_in))]
       
        cost =  cost_function(X, y, W_in, b_in, reg_lambda)

        if i % 10 == 0 or i == (num_iters-1):
            prediction_function(X, W_in, b_in, y)
            prediction_function(Xtest, W_in, b_in, Ytest)
            print(f"Iteration {i:4}: Cost {float(cost):8.2f}")

    return W_in, b_in

def predict(X, W, b, y):
    m, n = len(X), len(X[0])
    z = [[sum([X[i][k] * W[k][j] for k in range(n)]) + b[j] for j in range(len(b))] for i in range(m)]
    y_hat = softmax(z)
    p = [max(range(len(y_hat[i])), key=y_hat[i].__getitem__) for i in range(m)]
    correct = sum([1 if p[i] == y[i] else 0 for i in range(m)])
    accuracy =  correct / m
    print(f"Accuracy: , {accuracy * 100:.2f}%")
    return p

reg_lambda = 0
iterations = 100
alpha = 0.1

num_classes = 10
initial_W = [[0.01 * (random.random() - 0.5) for _ in range(num_classes)] for _ in range(n)]
initial_b = [0 for _ in range(num_classes)]

# Train the model on the X_train dataset
W, b = gradient_descent(X_test, Y_test, X_train, Y_train, initial_W, initial_b, compute_cost, compute_gradient, alpha, iterations, predict, reg_lambda)

# Test the model on the X_test dataset
print("Test set:")
predict(X_test, W, b, Y_test)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]

# Initialize an empty list to store accuracies for different learning rates
accuracies = []

# Loop over the learning rates
for learning_rate in learning_rates:
    # Train the model on the X_train dataset
    W, b = gradient_descent(X_test, Y_test, X_train, Y_train, initial_W, initial_b, compute_cost, compute_gradient, learning_rate, iterations, predict, reg_lambda)

    # Test the model on the X_test dataset
    accuracy = predict(X_test, W, b, Y_test)
    accuracies.append(accuracy)

# Plot the accuracy change based on the hyperparameter variations
plt.plot(learning_rates, accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Learning Rate')
plt.show()