import math
import random
import matplotlib.pyplot as plt

from minist.extract_features import MNISTFeatures


class LogisticRegression:
    def __init__(self, reg_lambda=0, iterations=100, alpha=0.1, num_classes=10):
        self.reg_lambda = reg_lambda
        self.iterations = iterations
        self.alpha = alpha
        self.num_classes = num_classes
        self.initial_W = None 
        self.initial_b = None
        
    def fit(self, X_train, Y_train):
        self.initial_W = [[0.01 * (random.random() - 0.5) for _ in range(self.num_classes)] for _ in range(len(X_train[0]))]
        self.initial_b = [0 for _ in range(self.num_classes)]
        self.W, self.b = self.gradient_descent(X_train, Y_train, self.initial_W, self.initial_b, self.compute_cost, self.compute_gradient, self.alpha, self.iterations)
        
    def predict(self, X_test, Y_test):
        p = self.predict_labels(X_test, self.W, self.b, Y_test)
        accuracy = self.calculate_accuracy(p, Y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return p
    
    def softmax(self, z):
        exp_z = [[math.exp(e) for e in row] for row in z]
        return [[e / sum(row) for e in row] for row in exp_z]
    
    def compute_cost(self, X, y, W, b):
        m, n = len(X), len(X[0])
        z = [[sum([X[i][k] * W[k][j] for k in range(n)]) + b[j] for j in range(len(b))] for i in range(m)]
        y_hat = self.softmax(z)
        y_one_hot = [[1 if c == y[i] else 0 for c in range(len(b))] for i in range(m)]
        total_cost = -(1/m) * sum([sum([y_one_hot[i][j] * math.log(y_hat[i][j]) for j in range(len(b))]) for i in range(m)])

        # Add L2 regularization term
        L2_reg = (self.reg_lambda / 2) * sum([sum([W[k][j]**2 for j in range(len(b))]) for k in range(n)])
        total_cost += L2_reg

        return total_cost
    
    def compute_gradient(self, X, y, W, b):
        m, n = len(X), len(X[0])
        z = [[sum([X[i][k] * W[k][j] for k in range(n)]) + b[j] for j in range(len(b))] for i in range(m)]
        y_hat = self.softmax(z)
        y_one_hot = [[1 if c == y[i] else 0 for c in range(len(b))] for i in range(m)]

        dj_db = [(1/m) * sum([y_hat[i][j] - y_one_hot[i][j] for i in range(m)]) for j in range(len(b))]
        dj_dW = [[(1/m) * sum([(y_hat[i][j] - y_one_hot[i][j]) * X[i][k] for i in range(m)]) for j in range(len(b))] for k in range(n)]

        # Add L2 regularization term for gradients
        dj_dW = [[dj_dW[k][j] + self.reg_lambda * W[k][j] for j in range(len(b))] for k in range(n)]
        return dj_db, dj_dW
    
    def gradient_descent(self, X, y, W_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        m = len(X)
        for i in range(num_iters):
            dj_db, dj_dW = gradient_function(X, y, W_in, b_in)

            # Update W and b
            W_out = [[W_in[k][j] - alpha * dj_dW[k][j] for j in range(len(b_in))] for k in range(len(W_in))]
            b_out = [b_in[j] - alpha * dj_db[j] for j in range(len(b_in))]

            # Compute cost for monitoring convergence
            cost = cost_function(X, y, W_out, b_out)
            if i % 100 == 0:
                print(f"Iteration {i}, cost={cost}")

            # Update W_in and b_in for next iteration
            W_in, b_in = W_out, b_out

        return W_out, b_out

    def predict_labels(self, X, W, b, Y):
        z = [[sum([X[i][k] * W[k][j] for k in range(len(X[0]))]) + b[j] for j in range(len(b))] for i in range(len(X))]
        return [row.index(max(row)) for row in z]
    
    def calculate_accuracy(self, p, y):
        correct = 0
        for i in range(len(p)):
            if p[i] == y[i]:
                correct += 1
        return correct / len(p)
    
