import re
import numpy as np
import random
import string
from math import sqrt



def textParser(text):
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    rwords = []
    for word in stripped:
        word = word.replace("\n", "")
        if len(word) > 2:
            rwords.append(word.lower())
    return " ".join(rwords)


def loadData(fileName):
    f = open(fileName)
    classLables = []
    commentLists = []
    for line in f.readlines():
        tag = line[8:16]
        if tag == 'True Pos':
            classLables.append(0)
        elif tag == 'Fake Pos':
            classLables.append(1)
        elif tag == 'True Neg':
            classLables.append(2)
        elif tag == 'Fake Neg':
            classLables.append(3)
        words = textParser(line[17:])
        commentLists.append(words)
    f.close()
    return commentLists, classLables

def loadData_test(fileName, fileName_key=None):
    classLables = []
    commentLists = []
    ids = []
    f = open(fileName)
    for line in f.readlines():
        id, text = line[:7], line[8:]
        words = textParser(text)
        commentLists.append(words)
        ids.append(id)
    f.close()

    if fileName_key:
        f2 = open(fileName_key)
        for line in f2.readlines():
            tag = line[8:16]
            if tag == 'True Pos':
                classLables.append(0)
            elif tag == 'Fake Pos':
                classLables.append(1)
            elif tag == 'True Neg':
                classLables.append(2)
            elif tag == 'Fake Neg':
                classLables.append(3)
        f2.close()
        return commentLists, classLables, ids
    return commentLists, ids

def sigmoid(out):
    return 1.0 / (1.0 + np.exp(-out))

def delta_sigmoid(out):
    return sigmoid(out) * (1 - sigmoid(out))

def SigmoidCrossEntropyLoss(a, y):
    return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))



class Percepton:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        classes = ['True Pos', 'Fake Pos', 'True Neg', 'Fake Neg']
        structure = [self.max_features, 25, len(classes)]

        self.biases = [np.random.randn(y, 1)/sqrt(y) for y in structure[1:]]
        self.weights = [np.random.randn(y, x)/sqrt(y) for x, y in zip(structure[:-1], structure[1:])]
        self.num_layers = len(structure)

    def one_hot_encode(self, classLabels):
        one_hot = np.zeros((len(classLabels), 4))
        one_hot[np.arange(len(classLabels)), classLabels] = 1
        return one_hot

    def feedforward(self, activation):
        activation = np.reshape(activation, [-1, len(activation)])
        activations = [activation]  # list to store activations for every layer
        outs = []  # list to store out vectors for every layer
        for b, w in zip(self.biases, self.weights):
            out = np.dot(w, activation) + b
            outs.append(out)
            activation = sigmoid(out)
            activations.append(activation)
        return outs, activations


    def backpropagate(self, x, y):
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        outs, activations = self.feedforward(x)
        y = np.transpose(y)
        loss = SigmoidCrossEntropyLoss(activations[-1], y)
        delta_cost = activations[-1] - y
        delta = delta_cost
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            out = outs[-l]
            delta_activation = delta_sigmoid(out)
            delta = np.dot(self.weights[-l + 1].T, delta) * delta_activation
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l - 1].T)
        return (loss, del_b, del_w)

    def train(self, X, y, batch_size=1, learning_rate=0.1, epochs=20, wordList={}, max_features=5000):
        n_batches = int(X.shape[0] / batch_size)
        train_len=X.shape[0]
        '''
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        filename = './dataset/dev-text.txt'
        filename2 = './dataset/dev-key.txt'
        commentLists_test, classLabels_test, ids = loadData_test(filename, filename2)
        counter_vectorizer = TfidfVectorizer(max_features=max_features, vocabulary=wordList, sublinear_tf=True)
        feature_vector_test = counter_vectorizer.fit_transform(commentLists_test).toarray()
        classLabels_test = self.one_hot_encode(classLabels_test)
        '''

        for j in range(epochs):
            randomlist = np.random.randint(0, train_len-batch_size, size=n_batches)

            batch_X = [X[k:k + batch_size] for k in randomlist]
            batch_y = [y[k:k + batch_size] for k in randomlist]

            for m in range(len(batch_y)):
                #del_b = [np.zeros(b.shape) for b in self.biases]
                #del_w = [np.zeros(w.shape) for w in self.weights]
                loss, delta_del_b, delta_del_w = self.backpropagate(batch_X[m], batch_y[m])
                #del_b = [db + ddb for db, ddb in zip(del_b, delta_del_b)]
                #del_w = [dw + ddw for dw, ddw in zip(del_w, delta_del_w)]

                self.weights = [w - (learning_rate / batch_size)
                            * delw for w, delw in zip(self.weights, delta_del_w)]
                self.biases = [b - (learning_rate / batch_size)
                           * delb for b, delb in zip(self.biases, delta_del_b)]

            learning_rate *= 0.95
            #print("Epoch %d complete\tLoss: %f" % (j, loss))
            #self.eval(X, y)
            #self.eval(feature_vector_test, classLabels_test)
            if epochs-2==j:
                model_para = [self.weights, self.biases, wordList]
                np.savetxt("averagedmodel.txt", model_para, fmt='%s')
                np.save("averagedmodel.txt.npy", model_para)


    def eval(self, X, y):
        count = 0
        for x, _y in zip(X, y):
            outs, activations = self.feedforward([x])
            predict_label = np.argmax(activations[-1])
            label = np.argmax(_y)
            if predict_label == label:
                count += 1
        print("\tAccuracy: %f" % ((float(count) / X.shape[0]) * 100))

    def predict(self, X):
        labels = {0:'True Pos', 1:'Fake Pos', 2:'True Neg', 3:'Fake Neg'}
        preds = np.array([])
        for x in X:
            outs, activations = self.feedforward([x])
            preds = np.append(preds, np.argmax(activations[-1]))
        preds = np.array([labels[int(p)] for p in preds])
        return preds