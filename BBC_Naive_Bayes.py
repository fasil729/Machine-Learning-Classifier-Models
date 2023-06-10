import matplotlib.pyplot as plt
from NaiveBayesMain import NaiveBayes, accuracy_score
from bbc.extract_fetures import Tfidf_method, split_data, BOW_method, MY_method,train_test_split



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


if __name__ == "__main__":
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
    plt.title('BBC datasets Classification')
    plt.legend()
    plt.show()