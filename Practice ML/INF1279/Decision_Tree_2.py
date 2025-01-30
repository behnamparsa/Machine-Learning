import re
import pandas as pd
import pickle

from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
import seaborn as sn
import numpy as np

#iris = load_iris(as_frame = True)
#target = iris['target']
#df = iris['data']
#print(df.head())
#print()

#titanic = pd.read_csv('titanic.csv')
#print(titanic.head())
#print()

df = pd.read_pickle('titanic.pickle')
features = df.drop('alive', axis=1)
target = df['alive']
print(features.head())
print()
print(target.head())

features_encoded = pd.get_dummies(features, drop_first=True)

print(features_encoded.head())

X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, shuffle=True, train_size=0.7)

print(X_train.shape)
print(X_test.shape)
#---- fit the model ------------------------------

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)
y_predicted_train = classifier.predict(X_train)
y_predicted_test = classifier.predict(X_test)

# ------------- accuracy test ---------------------
print("Train Accuracy: ", accuracy_score(y_train, y_predicted_train))  
print("Test Accuracy: ", accuracy_score(y_test, y_predicted_test))


#------------ tuning the parameters ---------------------------------------


best_acc = 0

for criterion in ['gini','entropy']:
    for max_depth in [2, 3, 4, 5, 6]:
        for min_sample_leaf in [5, 10, 20, 30]:
            classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_sample_leaf)
            classifier.fit(X_train, y_train)
            y_predicted_test = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_predicted_test)
            if acc > best_acc:
                best_acc = acc
                best_criterion = criterion
                best_max_depth = max_depth
                best_min_sample_leaf = min_sample_leaf
                best_params = f"criterion: {best_criterion}, max_depth: {best_max_depth}, min_sample_leaf: {best_min_sample_leaf}"

print(best_params)
print(best_acc)


# plot the tree
fig = plt.figure(figsize=(25,20))
plot_tree(classifier,
            feature_names=features_encoded.columns,
            class_names=target.unique(),
            impurity=False,
            proportion=True,
            filled=True)
fig.savefig(r'C:\Users\Parsazh\Documents\GitHub\Machine-Learning\Practice ML\INF1279\titanic_tree.png')


#Plot Tree by Seaborn

fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize=(16,9))

axes[0].set_title("True")
axes[1].set_title("Predicted")

sn.scatterplot(data = X_test,x = 'who_man', y = 'who_woman', hue = y_test, ax = axes[0], palette = 'viridis')
sn.scatterplot(data = X_test,x = 'who_man', y = 'who_woman', hue = y_predicted_test, ax = axes[1], palette = 'viridis')
plt.show()
