import numpy as np
from percepton3 import Percepton
from percepton3 import loadData
from percepton3 import loadData_test
import sys


if __name__=="__main__":
    filename = "./dataset/train-labeled.txt"
    #filename = sys.argv[1]
    max_features = 7800
    commentLists, classLabels = loadData(filename)
    from sklearn.feature_extraction.text import TfidfVectorizer
    counter_vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    feature_vector = counter_vectorizer.fit_transform(commentLists).toarray()
    vocabularyList = counter_vectorizer.vocabulary_
    model = Percepton(max_features=len(feature_vector[0]))

    '''
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    X_indices = np.arange(feature_vector.shape[-1])
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
            edgecolor='black')
    '''
    classLabels = model.one_hot_encode(classLabels)

    model.train(feature_vector, classLabels, epochs=15, wordList=vocabularyList, max_features=len(vocabularyList))

    model_para = [model.weights, model.biases, vocabularyList]
    np.savetxt("vanillamodel.txt", model_para, fmt='%s')
    np.save("vanillamodel.txt.npy", model_para)

