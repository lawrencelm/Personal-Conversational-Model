import numpy as np
import pyttsx
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#from samr.transformations import POSTagger
#from samr.transformations import (ExtractText, ReplaceText, MapToSynsets, POSTagger, SentimentChangerTagger,
#                                  Polarity_And_Subjectivity, Densifier, ClassifierOvOAsFeatures)
engine = pyttsx.init()
engine.setProperty('rate', 100)
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)

fname = "all.txt"
with open(fname) as f:
    replies = []
    for reply in f.readlines():
        if len(reply) > 1 and (':' in reply.split()[0]):
            words = reply.split()[1:]
			# print words
            new_words = []
            for word in words:
                if "..." in word:
                    word = word.replace("...","")
                if ".." in word:
                    word = word.replace("..","")
                if "(" in word:
                    word = word.replace("(","")
                if ")" in word:
                    word = word.replace(")","")
                if "?" in word:
                    word = word.replace("?", " ")
                if "[" in word and "]" in word:
                    ind2 = word.find("]")
                    ind1 = word.find("[")
                    word = word[:ind1] + word[ind2:]
                if "]" in word:
                    word = word.replace("]", " ")
                if "[" in word:
                    word = word.replace("[", " ")
                if "." in word:
                    word = word.replace(".", " ")

                if len(word) != 0:
                    last_ch = word[len(word)-1]
                    if last_ch is "." or last_ch is "!" or last_ch is "?" or last_ch is ";" or last_ch is ",":
                        word = word[:len(word)-1]
                    new_words.append(word)
			# print " ".join(new_words)
            replies.append(" ".join(new_words) + " <EOS>")
    # replies = [" ".join(reply.split()[1:]) + " <EOS>" for reply in f.readlines()]
replies = [reply.lower() for reply in replies if reply != ""]
labels = []
freplies = []
print replies 

for i in range(len(replies)-1):
	reply = replies[i]
	# reply = reply[:len(reply)-6]
	freplies.append(reply)
	words = replies[i+1].split()
	labels.append(words[0])
	for j in range(len(words)-1):
		word = words[j]
		reply = reply + " " + word
		freplies.append(reply)
		labels.append(words[j+1])

# for i in range(len(replies)-1):
# 	reply = replies[i+1]
# 	labels.append(reply.split()[0])
# labels.append("end")
# print freplies
# print labels

X_train = np.array(freplies)
#y_train = np.matrix('"hi";0;0;0;0;0;1;1;1;1;1;1;1;1')
y_train = np.array(labels)
#y_train = [[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0,1],[0,1]]
X_test = np.array(['hello', 
                   'how are you ?',
                   'bye', 
					"why are we here",
					"when were you born?",
					"see you", 
					"good ",
					"what is my name",
					"not", "july"])   

classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=tuple([1,2]))),
    ('tfidf', TfidfTransformer()),

#    ('pos', POSTagger()),
    #('clf', MultinomialNB(alpha=0, class_prior=None, fit_prior=True))])
    #('clf', svm.SVC(decision_function_shape='ovo'))]) #1 - bad
    #('clf', svm.LinearSVC())]) #2 - good
    # ('clf', OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0))])
    # ('clf', LogisticRegression(solver="newton-cg", multi_class="ovr"))]) #4 - good
    ('clf', OneVsRestClassifier(LinearSVC()))]) #not that good
    #('clf',  MultinomialNB(alpha=0, class_prior=None, fit_prior=True))]) - decent
    #('clf', SGDClassifier())]) #5 

for i in range(X_train.size):
    print str(X_train[i]) + " >>>> " + str(y_train[i])

classifier.fit(X_train, y_train)

#print replies
#predicted = classifier.predict(X_test)
#print predicted
# os.system("voice " + "Victoria")
os.system("echo " + "I am done")
os.system("say " + "I am done")
while True:
    utterance = raw_input("Tell me: ")
    if utterance == "":
        break
    utterance += ' <eos>'
    response = ""
    numIt = 0
    not_reply_end = True
    lastWord = ""
    while not_reply_end:
        #print utterance
        nextWord = classifier.predict([utterance])
        print "Predicted: ", nextWord
        # print "Next word: ", nextWord
        # print nextWord[0][len(nextWord)-6:len(nextWord)-1]
        # end = nextWord[0][len(nextWord[0])-5:len(nextWord[0])]
        if nextWord[0] == '<eos>':
            os.system("echo " + response)
            os.system("say " + response)
            nextWord[0] = nextWord[0][:len(nextWord[0])-5]
            not_reply_end = False
        elif nextWord[0] == lastWord:
            os.system("echo " + response)
            os.system("say " + response)
            not_reply_end = False
        print "predicting.." + str(nextWord)
        response = response + " " + nextWord[0]

        nextSequence = utterance + " " + nextWord[0]
        #print nextSequence
        utterance = nextSequence
        numIt = numIt + 1
        lastWord = nextWord[0]
# engine.runAndWait()
    print ">>>>> " + response
for item, label in zip(freplies, labels):
    print '%s => %s' % (item, label)# for x in labels))

# for item, label in zip(X_test, predicted):
#     print '%s => %s' % (item, label)# for x in labels))
