import numpy as np
import math
import os
# from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def RemoveStopwords(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = [x for x in tokens if not x in stop_words]
    return " ".join(filtered_words)

def RemoveHeader(words, type):
    if('lines' in words):
        index_lines = words.index("lines")
        index_lines = index_lines + 2
        result = words[index_lines:]
        if (type==0):
            TrainingInstance.append(result)
        else:
            TestingInstance.append(result)
    else:
        if (type==0):
            TrainingInstance.append(words)
        else:
            TestingInstance.append(words)

def ReadFile(path, type):
    # print(type)
    for filename in os.listdir(path):
        # print('Appending ClassLabel',ClassLabels)
        innerpath = path + '/' + filename
        for innerfilename in os.listdir(innerpath):
            if (type == 0):
                ClassLabelsTrain.append(filename)
            else:
                ClassLabelsTest.append(filename)
            innermostpath = innerpath + '/' + innerfilename
            file = open(innermostpath)
            text = file.read()
            words = RemoveStopwords(text).split()
            RemoveHeader(words,type)

def ExtVocab(TrainingInstances):
    TotalTrainingInstances = len(TrainingInstances);
    # print(TotalTrainingInstances)
    Vocab = []
    for i in range(TotalTrainingInstances):
        Vocab = Vocab + TrainingInstances[i]
    Vocab = list(set(Vocab))
    # print(Vocab)
    return Vocab

def CountOfDocuments(TrainingInstances):
    return len(TrainingInstances)

def GetClassLabels(ClassLabels):
    return list(set(ClassLabels))

def CountOfDocumentsInClass(ClassLabels, Class):
    return ClassLabels.count(Class)

def ConcatDocumentsOfClass(TrainingInstances, ClassLabels, Class):
    TextOfClass = []
    for i in range(len(TrainingInstances)):
        if(ClassLabels[i]==Class):
            for j in range(len(TrainingInstances[i])):
                TextOfClass.append(TrainingInstances[i][j])
    return TextOfClass

def CountTokensOfTerm(TextOfClass, Text):
    return TextOfClass.count(Text)

def CountOfFeaturesInClass(ClassLabel, TrainingInstances, ClassLabels):
    count= 0
    for i in range(len(TrainingInstances)):
        if(ClassLabels[i] == ClassLabel):
            count = count + len(TrainingInstances[i])
    return count

def GetTokensFromDocument(Vocab,d):
    W=[]
    for i in range(len(d)):
        t=d[i]
        # print(t)
        if(t in Vocab):
            # print("true")
            W.append(t)
        # else:
    return W

def IndexOfTermInV(t,Vocab):
    return Vocab.index(t)

def TrainMultinomialNaiveBayes(C, D):
    # print('In the train MNB function')
    Vocab = ExtVocab(D)
    # print('Vocab', Vocab)
    N = CountOfDocuments(D)
    # print('Count of training Examples', N)
    Prior = []
    ProbabilityConditional = np.zeros((len(Vocab), len(C)))
    # print(ProbabilityConditional)
    B = len(Vocab)
    for i in range(len(C)):
        c = C[i]
        # print('Current Class', c)
        NC = CountOfDocumentsInClass(ClassLabelsTrain, c)
        # print(NC)
        Prior.append(NC / N)
        # print('Prior ', Prior[i])
        ConcatenatedText = ConcatDocumentsOfClass(D, ClassLabelsTrain, c)
        # print('Concatenated Text',ConcatenatedText)
        AllFeatures = CountOfFeaturesInClass(c, D, ClassLabelsTrain)
        # print('All Features: ',AllFeatures)
        for x in range(len(Vocab)):
            t = Vocab[x]
            # print('Token', t)
            TokenCount = CountTokensOfTerm(ConcatenatedText, t)
            # print('Count Of Tokens ', TokenCount)
            ProbabilityConditional[x][i] = (TokenCount + 1) / (AllFeatures + B)
    # print('Result')
    # print('Vocabulary: ', Vocab)
    # print('Prior: ', Prior)
    # print('Conditional Probability: ', ProbabilityConditional)
    # UseNaiveBayes(C, Vocab, Prior, ProbabilityConditional, D)
    return Vocab, Prior, ProbabilityConditional

def UseNaiveBayes(C, Vocab, Prior, ProbabilityConditional, TestingInstances):
    W = GetTokensFromDocument(Vocab, TestingInstances)
    # print('Tokens',W)
    score=[]
    # print('Class Label',len(C));
    for i in range(len(C)):
        c = C[i]
        # print('Current Class: ', c)
        score.append(math.log(Prior[i]))
        for x in range(len(W)):
            t = W[x]
            y = IndexOfTermInV(t,Vocab)
            score[i] = score[i]+ math.log(ProbabilityConditional[y][i])
        # print(c, score[i])
    IndexPrediction = np.argmax(score)
    Prediction = C[IndexPrediction]
    # print('Prediction is :', Prediction)
    return Prediction

def CalculateAccuracy(Vocab, Prior, ProbabilityConditional, C, ClassLabelsTest, TestingInstance ):
    AccurateCount = 0
    # print('length of test instance',len(TestingInstance))
    for i in range(len(TestingInstance)):
        # print('loop variable', i)
        d = TestingInstance[i]
        Prediction = UseNaiveBayes(C, Vocab, Prior, ProbabilityConditional, d)
        if(Prediction == ClassLabelsTest[i]):
            AccurateCount = AccurateCount +1
    return (AccurateCount/len(TestingInstance))*100

if __name__ == "__main__":
    # TestFlag = 0
    TrainingPath = input('Enter the Training Path: ')
    TestingPath = input('Enter the Testing Path: ')
    # TrainingPath = "D:/Fall'17/MachineLearning/20news-bydate/20news-bydate-train"
    # TestingPath = "D:/Fall'17/MachineLearning/20news-bydate/20news-bydate-test"
    # TrainingInstance = [['Chinese', 'Beijing', 'Chinese'],
    #                      ['Chinese', 'Chinese', 'Shanghai'],
    #                      ['Chinese', 'Macao'],
    #                      ['tokyo', 'Japan', 'Chinese']]
    # TestingInstance = [['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']]
    # ClassLabelsTrain=['yes', 'yes', 'yes', 'no']
    # ClassLabelsTest=[]
    TrainingInstance = []
    TestingInstance = []
    ClassLabelsTrain = []
    ClassLabelsTest = []
    ReadFile(TrainingPath, 0)
    ReadFile(TestingPath, 1)
    print("Length of Training Instance: ", len(TrainingInstance) )
    # print(len(TrainingInstance))
    # print(TrainingInstance)
    print("Number of Training Class Labels: ", len(ClassLabelsTrain))
    # print(len(ClassLabelsTrain))
    # print(ClassLabelsTrain)
    print("Length of Testing Instance: ", len(TestingInstance))
    # print(len(TestingInstance))
    # print(TestingInstance)
    print("Number of Testing Class Labels: ", len(ClassLabelsTest))
    # print(len(ClassLabelsTest))
    # print(ClassLabelsTest)
    Vocab, Prior, ProbabilityConditional=TrainMultinomialNaiveBayes(ClassLabelsTrain, TrainingInstance)
    # print(Vocab, Prior, ProbabilityConditional)
    # UseNaiveBayes(ClassLabelsTrain, Vocab, Prior, ProbabilityConditional, TestingInstance)
    Accuracy = CalculateAccuracy(Vocab, Prior, ProbabilityConditional, ClassLabelsTrain, ClassLabelsTest, TestingInstance)
    print('Accuracy is: '+str(Accuracy)+'%')