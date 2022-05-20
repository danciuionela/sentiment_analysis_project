import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features) 
            votes.append(v)
        return mode(votes) #returning the most popular vote

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features) 
            votes.append(v)

        choice_votes = votes.count(mode(votes)) 
        conf = choice_votes / len(votes) #distribution scale is from 0 to 1 
        return conf #returning confidence indicator


# opening frequent words pickle from file
word_features5k_f = open("algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


# generating a dictionary of features
#(keys are words and values are boolean values as True if word exists in doc)
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# opening classifier pickles from file, and performing classification

open_file = open("algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats) #classifier - pos/neg + confidence level
   