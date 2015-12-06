import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])
#y_train = np.matrix('"hi";0;0;0;0;0;1;1;1;1;1;1;1;1')
y_train = np.array(["hi","hi","hi","hi","hi","hi","bye","bye","bye","bye","bye","bye","bye","bye"])
#y_train = [[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0,1],[0,1]]
X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'hello welcome to new york. enjoy it here and london too'])   
target_names = ['New York', 'London']

classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=tuple([1,2]))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print predicted
for item, labels in zip(X_test, predicted):
    print '%s => %s' % (item, ', '.join(target_names[labels]))# for x in labels))
