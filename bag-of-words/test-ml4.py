import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

X_train = np.array(["hello",
                    "how are you",
                    "how are you I",
                    "how are you I am",
                    "what is your name",
                    "what is your name I",
                    "what is your name I am",
                    "when were you born",
                    "when were you born july",
                    "see you later",
                    "I see what you treat me now",
                    "what is the name of my mother"])
y_train = np.array(["hello","I","am","good", "I", "am", "Julia", "july", "20th", "bye", "whoops", "francesca"])
X_test = np.array(['hello',
                   'how are you doing',
                   'what is my name',
                   'how are you doing today',
                   'how are you I am good',
                   'what is your name I am',
                   'hello jerry',
                   'how are you today',
                   'how are you today I',
                   'how are you today I am',
                   'great to hear that!',
                   'may I ask you what your name is',
                   'may I ask you what your name is I',
                   'may I ask you what your name is I am',
                   'and in what day and month were you born',
                   'and in what day and month were you born july',
                   'see you',
                   'I will see you on wednesday then',
                   'I see what you did there',
                   'tell me how your mother is called',
                   'tell me the name of your mother dude',
                   'what is your name',
                   'what is your freaking name'])   

classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=tuple([1,2]))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, y_train)
print classifier.predict(["how how are you"])
predicted = classifier.predict(X_test)
print predicted
