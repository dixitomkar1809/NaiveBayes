'''with open('20news-bydate/20news-bydate-train/comp.graphics/37261') as f:
    start = False
    print(start)
    for line in f:
        if "Lines" in line:
            start = True
            print(start)
        if start:  # if True we have found the section we want
            for line in f:
                print(line)'''
import io
import os
import re
import string
import nltk
#from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
# from MNB import *

stop_words = set(stopwords.words('english'))
#stop_words = get_stop_words('english')
# path = "D:/Fall'17/MachineLearning/20news-bydate/20news-bydate-train"


def remove_stopwords(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = [x for x in tokens if not x in stop_words]
    return " ".join(filtered_words)


def remove_header(words, type):
    if('lines' in words):
        # print('lines is in words')
        index_lines = words.index("lines")
        index_lines = index_lines + 2
        result = words[index_lines:]
        # return result
        if (type==0):
            TrainingInstance.append(result)
            # print(TrainingInstance)
        else:
            TestingInstance.append(result)
            # print(TestingInstance)
        # print(result)
    else:
        # print('lines not in words')
        print(words)
    # print('-----------------------------------------------------------------Execution of One File-----------------------------------------------------------------')

def ReadFile(path, type):
    # print(type)
    for filename in os.listdir(path):
        ClassLabels.append(filename)
        innerpath = path + '/' + filename
        for innerfilename in os.listdir(innerpath):
            innermostpath = innerpath + '/' + innerfilename
            file = open(innermostpath)
            text = file.read()
            words = remove_stopwords(text).split()
            remove_header(words,type)

if __name__ == "__main__":
    TestFlag=0
    TrainingPath = "D:/Fall'17/MachineLearning/20news-bydate/20news-bydate-train"
    TestingPath = "D:/Fall'17/MachineLearning/20news-bydate/20news-bydate-test"
    TrainingInstance = []
    TestingInstance = []
    ClassLabels=[]
    ReadFile(TrainingPath)
    # print(TrainingInstance)
    # print(ClassLabels)
    ReadFile(TestingPath)




