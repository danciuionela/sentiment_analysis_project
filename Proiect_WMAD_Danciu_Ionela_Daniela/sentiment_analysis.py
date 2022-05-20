import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize


short_pos = open("classified_reviews/positive.txt", "r").read()
short_neg = open("classified_reviews/negative.txt", "r").read()

#array of tuples for review classification as pos/neg
all_words = [] #only the words that respects the allowed_word_types J,V,R
documents = [] #all words from both files

# allowed_word_types: J adjective, R adverb, V verb
allowed_word_types = ["J","R","V"] 

for p in short_pos.split('\n'):

    # generate tuples of review - label pairs (word, "pos"/"neg")
    documents.append((p, "pos")) 

    # tokenization to create a BOW model
    words = word_tokenize(p) #splits the sentence in words

    pos = nltk.pos_tag(words) #function from nltk which assigns a speech tag NN - noun, RB - adverb, JJ etc

    for w in pos: #for every tuple above
        # conversion to lowercase
        if w[1][0] in allowed_word_types: #if the first letter from the second word is in the allowes types: J,V,R
            all_words.append(w[0].lower()) #add word in all_words


for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


#saving as pickle (binary format) the words in the documents.pickle
save_documents = open("algos/documents.pickle", "wb") 
pickle.dump(documents, save_documents) #dumb for serialization 
save_documents.close()

# taking words and generating a frequency distribution
all_words = nltk.FreqDist(all_words) 

word_features = list(all_words.keys())[:5000] #list with only the keys = words (the values are the numbers of appearance)

# saving the most frequent words as a pickle
save_word_features = open("algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features) 
save_word_features.close()


#READ FROM BINARY FILE
# opening frequent words pickle from file
# infile = open("algos/word_features5k.pickle",'rb')
# new_dict = pickle.load(infile)
# infile.close()
# print(new_dict)


# print(word_features)

# generating a dictionary of features
#(keys are words and values are boolean values as True if word exists in doc)

def find_features(document):
    words = word_tokenize(document) #split in words
    features = {} #define dictionary 
    for w in word_features: #word features contains words of types J,R,V classified from txt file
        features[w] = (w in words) #for every word from word_features looks if the word is in words => True/False
    return features

# creating a features vector for each review (feature, "pos"/"neg")
featuresets = [(find_features(rev), category) for (rev, category) in documents] 
# print(featuresets)

random.shuffle(featuresets)
print("Lungime set de date: ", len(featuresets)) #10690

#defining the train-test sets
training_set = featuresets[:10000] 
testing_set = featuresets[10000:] 

print("Using classifiers using {} training samples and {} testing samples...\n".format(len(training_set), len(testing_set)))


# performing classification using various classifiers

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100) 
classifier.show_most_informative_features(15) 


save_classifier = open("algos/originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("algos/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=10690))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("algos/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

save_classifier = open("algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

