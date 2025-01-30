import re
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame = True)
target = iris['target']
df = iris['data']

X_train, X_test, y_train, y_test = train_test_split(df, target, shuffle=True, train_size=0.7)

Classifier = DecisionTreeClassifier()
Classifier.fit(X_train, y_train)

y_predicted = Classifier.predict(X_test)
print(accuracy_score(y_test, y_predicted))
