import numpy as np
from percepton3 import Percepton
from percepton3 import loadData
from percepton3 import loadData_test
import sys
from preProcess import Preprocess

if __name__=="__main__":



    filename = "./dataset/train-labeled.txt"
    #filename = sys.argv[1]
    filename = "./dataset/train-labeled.txt"
    # filename = sys.argv[1]
    pt = Preprocess()
    commentLists, classLabels = pt.loadData(filename)
    vocabularyList = pt.createVocabularyList(commentLists)
    trainLists = pt.bagOfWords(vocabularyList, commentLists)
    feature_vector = np.array(trainLists)
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    commentLists, classLabels = loadData(filename)
    counter_vectorizer = TfidfVectorizer(max_features=model.max_features)
    feature_vector = counter_vectorizer.fit_transform(commentLists).toarray()
    classLabels = model.one_hot_encode(classLabels)
    wordList = list(counter_vectorizer.vocabulary_)
    '''
    model = Percepton(max_features=len(vocabularyList))
    classLabels = model.one_hot_encode(classLabels)

    model.train(feature_vector, classLabels, epochs=1000, wordList=vocabularyList, max_features=len(vocabularyList))



    model_para = [model.weights, model.biases, vocabularyList]
    np.savetxt("vanillamodel.txt", model_para, fmt='%s')
    np.save("vanillamodel.txt.npy", model_para)

