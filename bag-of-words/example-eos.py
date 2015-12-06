import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

fname = "friends-lexicon.txt"
with open(fname) as f:
    replies = [" ".join(reply.split()[1:]) + " <EOS>" for reply in f.readlines()]
replies = [reply.lower() for reply in replies if reply != ""]
print replies
labels = []
freplies = []
# print replies 

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
    ('vectorizer', CountVectorizer(ngram_range=tuple([1,3]))),
    ('tfidf', TfidfTransformer()),
    ('clf',  MultinomialNB(alpha=0, class_prior=None, fit_prior=True))])

classifier.fit(X_train, y_train)
print replies
predicted = classifier.predict(X_test)
print predicted
while True:
    utterance = raw_input()
    if utterance == "":
        break
    response = ""
    numIt = 0
    not_reply_end = True
    while not_reply_end:
        #print utterance
        nextWord = classifier.predict([utterance])
        # print nextWord[0][len(nextWord)-6:len(nextWord)-1]
        # end = nextWord[0][len(nextWord[0])-5:len(nextWord[0])]
        if nextWord[0] == '<eos>':
        	# nextWord[0] = nextWord[0][:len(nextWord[0])-5]
        	not_reply_end = False
        #print nextWord
        response = response + " " + nextWord[0]
        nextSequence = utterance + " " + nextWord[0]
        #print nextSequence
        utterance = nextSequence
        numIt = numIt + 1
    print ">>>>> " + response
for item, label in zip(freplies, labels):
    print '%s => %s' % (item, label)# for x in labels))
# for item, label in zip(X_test, predicted):
#     print '%s => %s' % (item, label)# for x in labels))
