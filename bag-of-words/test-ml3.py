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
                    "what is your name I am"])
y_train = np.array(["hello","I","am","good", "I", "am", "Julia"])
X_test = np.array(['hello',
                   'how are you doing',
                   'what is my name',
                   'how are you doing today',
                   'how are you I am good',
                   'what is your name I am'])   

classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=tuple([1,2]))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(fit_prior=True))])#OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print predicted
