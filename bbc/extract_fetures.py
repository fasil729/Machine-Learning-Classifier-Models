from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import coo_matrix

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class BOW_method:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder

    def extract_top_k_features(self, X, y, k):
        # Extract top k features for each class using BoW
        vectorizer = CountVectorizer()
        feature_names = []
        feature_scores = np.asarray(X.sum(axis=0)).ravel()
        top_k_indices = feature_scores.argsort()[::-1][:k]
        # feature_names.extend(vectorizer.get_feature_names_out()[top_k_indices])
        # feature_names = list(set(feature_names))
        # print(feature_names)
        return X[:, top_k_indices], y


    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        corpus = [(int(doc)-1, int(term)-1, int(float(freq))) for term, doc, freq in (line.split() for line in matrix)]
        doc_ids, term_ids, freqs = zip(*corpus)
        X = np.zeros((len(docs), len(terms)))
        for doc_id, term_id, freq in corpus:
            X[doc_id, term_id] = freq
       
        y = [int(line.split()[1]) for line in classes]

        return X, y, docs
    
class MY_method:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder

    def extract_top_k_features(self, X, y, k):
        
        n_features = len(X[0])
        max_feature_freq = np.zeros(n_features)
        
        for i in range(n_features):
            feature_freq = []
            for c in set(y):
                class_indices = np.where(y == c)[0]
                class_feature_counts = X[class_indices, i].sum()
                feature_freq.append(class_feature_counts)
            max_feature_freq[i] = max(feature_freq)
        
        feature_sums = np.asarray(X.sum(axis=0)).ravel()
        max_feature_freq /= feature_sums
        
        top_k_indices = max_feature_freq.argsort()[::-1][:k]
    
        return X[:, top_k_indices], y


    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        corpus = [(int(doc)-1, int(term)-1, int(float(freq))) for term, doc, freq in (line.split() for line in matrix)]
        doc_ids, term_ids, freqs = zip(*corpus)
        X = np.zeros((len(docs), len(terms)))
        for doc_id, term_id, freq in corpus:
            X[doc_id, term_id] = freq
       
        y = [int(line.split()[1]) for line in classes]

        return X, y, docs

class Tfidf_method:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder
    
    def extract_top_k_features(self, X, y, k, data):
        # Extract top k features for each class using TF-IDF
        feature_names = []
        vectorizer = TfidfVectorizer(norm=None, use_idf=True, preprocessor=self.extract_freq_count)
        X_class_tfidf = vectorizer.fit_transform(data)
        feature_scores = np.asarray(X_class_tfidf.sum(axis=0)).ravel()
        top_k_indices = feature_scores.argsort()[::-1][:k]
        feature_names.extend(vectorizer.get_feature_names_out()[top_k_indices])
        feature_names = list(set(feature_names))
        return X[:, top_k_indices], y

    def extract_freq_count(self, doc_tuple):
        return ' '.join([term + ' ' + str(freq) for term, freq in doc_tuple[2].items()])

    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        # Preprocess the data
        corpus_dict = {}
        for i, line in enumerate(matrix):
            term_id, doc_id, freq = line.split()
            doc_name = f"docs{doc_id}"
            term_idx = int(term_id) - 1  # convert to 0-based index
            term = terms[term_idx]
            freq = int(float(freq))
            if freq >= 0:
                if doc_name not in corpus_dict:
                    corpus_dict[doc_name] = {}
                corpus_dict[doc_name][term] = freq

        data = []
        for i, (doc_name, freq_dict) in enumerate(corpus_dict.items()):
            doc_index = i
            data.append((doc_name, doc_index, freq_dict))

        # Create a sparse matrix from the data
        corpus = [(int(doc)-1, int(term)-1, int(float(freq))) for term, doc, freq in (line.split() for line in matrix)]
        doc_ids, term_ids, freqs = zip(*corpus)
        X = np.zeros((len(docs), len(terms)))
        for doc_id, term_id, freq in corpus:
            X[doc_id, term_id] = freq

        y = [int(line.split()[1]) for line in classes]

        return X, y, data

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train):
    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

def test_model(clf, X_test, y_test):
    # Test the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')





if __name__ == '__main__':
    tfidf = Tfidf_method()
    X, y, data = tfidf.preprocess_data()
    X, y = tfidf.extract_top_k_features(X, y, 2000, data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # X_train_features, X_test_features = extract_features(X_train, X_test, y_train, k=3)
    clf = train_model(X_train, y_train)
    print("using tfidf method")
    test_model(clf, X_test, y_test)

    bow = BOW_method()
    X, y, data = bow.preprocess_data()
    X, y = bow.extract_top_k_features(X, y, 150)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train, X_test)
    # X_train_features, X_test_features = extract_features(X_train, X_test, y_train, k=3)
    clf = train_model(X_train, y_train)
    print("using bow method")
    test_model(clf, X_test, y_test)




    my = MY_method()
    X, y, data = my.preprocess_data()
    X, y = my.extract_top_k_features(X, y, 1500)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train, X_test)
    # X_train_features, X_test_features = extract_features(X_train, X_test, y_train, k=3)
    clf = train_model(X_train, y_train)
    print("using my method")
    test_model(clf, X_test, y_test)





# from sklearn.model_selection import train_test_split
# import numpy as np

# X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
# y = np.array([0, 0, 1, 1, 1])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Training features: \n", X_train)
# print("Training targets: \n", y_train)
# print("Testing features: \n", X_test)
# print("Testing targets: \n", y_test)