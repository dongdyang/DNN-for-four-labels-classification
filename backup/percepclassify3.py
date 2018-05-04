import numpy as np
from percepton3 import Percepton
from percepton3 import loadData
from percepton3 import loadData_test
import sys

if __name__=="__main__":

    #filename = sys.argv[2]
    filename = './dataset/dev-text.txt'
    filename2 = './dataset/dev-key.txt'
    filename3 = 'percepoutput.txt'
    model = Percepton()
    model_para = np.load('vanillamodel.txt.npy')
    model.weights, model.biases, wordList = model_para[0], model_para[1], model_para[2]
    from sklearn.feature_extraction.text import TfidfVectorizer

    commentLists_test, classLabels_test, ids = loadData_test(filename, filename2)
    classLabels_test = model.one_hot_encode(classLabels_test)
    #commentLists_test, ids = loadData_test(filename)

    counter_vectorizer = TfidfVectorizer(max_features=model.max_features, vocabulary=wordList)
    feature_vector_test = counter_vectorizer.fit_transform(commentLists_test).toarray()

    model.eval(commentLists_test, classLabels_test)

    preds = model.predict(feature_vector_test)
    fout = open(filename3, 'w')
    for id, pred in zip(ids, preds):
        fout.write(id+' '+pred+'\n')
    fout.close()

