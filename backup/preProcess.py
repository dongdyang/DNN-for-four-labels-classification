import string

class Preprocess:
    def textParser(self, text):
        #words = re.split(r'\W*', text)
        #regEx = re.compile(r'[^a-zA-Z]|\d')
        #words = regEx.split(text)
        words = text.split()
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        rwords = []
        for word in stripped:
            word = word.replace("\n", "")
            if len(word) > 1:
                rwords.append(word.lower())
        return rwords

    def loadData(self, fileName):
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
            words = self.textParser(line[17:])
            commentLists.append(words)
        return commentLists, classLables

    def createVocabularyList(self, commentLists):
        vocabularySet = set([])
        for comment in commentLists:
            vocabularySet = vocabularySet | set(comment)
        vocabularySet = vocabularySet
        vocabularyList = list(vocabularySet)
        return vocabularyList

    def commentListToVector(self, vocabularyList, commentList, pos_neg):
        vocabMarked = [0] * len(vocabularyList)
        for ele in commentList:
            if ele in vocabularyList:
                index = vocabularyList.index(ele)
                vocabMarked[index] += 2
                #if pos_neg and (ele in self.pos or ele in self.neg):
                #   vocabMarked[index] += 0

        return vocabMarked

    def bagOfWords(self, vocabularyList, commentLists, pos_neg=0):
        vocabMarkedList = []
        for i in range(len(commentLists)):
            vocabMarked = self.commentListToVector(vocabularyList, commentLists[i], pos_neg)
            vocabMarkedList.append(vocabMarked)
        return vocabMarkedList

