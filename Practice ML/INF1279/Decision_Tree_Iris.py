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

iris = load_iris(as_frame = True)
target = np.choose(iris['target'],iris['target_names'])
df = iris['data']
df.rename(columns = {'sepal length (cm)':'sepal_length', 'sepal width (cm)':'sepal_width', 'petal length (cm)':'petal_length', 'petal width (cm)':'petal_width'}, inplace = True)
features = df

X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=True, train_size=0.7)

print(X_train.shape)
print(X_test.shape)
#---- fit the model ------------------------------

#classifier = DecisionTreeClassifier(max_depth = 3)
#classifier.fit(X_train, y_train)
#y_predicted_train = classifier.predict(X_train)
#y_predicted_test = classifier.predict(X_test)

# ------------- accuracy test ---------------------
#print("Train Accuracy: ", accuracy_score(y_train, y_predicted_train))  
#print("Test Accuracy: ", accuracy_score(y_test, y_predicted_test))


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
#---- fit the model with the best paramenters ------------------------------

classifier = DecisionTreeClassifier(max_depth = best_max_depth, criterion = best_criterion, min_samples_leaf = best_min_sample_leaf)
classifier.fit(X_train, y_train)
y_predicted_train = classifier.predict(X_train)
y_predicted_test = classifier.predict(X_test)

# ------------- accuracy test ---------------------

print("Train_best Accuracy: ", accuracy_score(y_train, y_predicted_train))  
print("Test_best Accuracy: ", accuracy_score(y_test, y_predicted_test))

# plot the tree
fig = plt.figure(figsize=(16,9))
plot_tree(classifier,
            feature_names=features.columns,
            #class_names=target.unique(),
            impurity=False,
            proportion=True,
            filled=True)
#fig.savefig(r'C:\Users\Parsazh\Documents\GitHub\Machine-Learning\Practice ML\INF1279\titanic_tree.png')


#Plot Tree by Seaborn

fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize=(16,9))

axes[0].set_title("True")
axes[1].set_title("Predicted")

sn.scatterplot(data = X_test,x = 'petal_length', y = 'petal_width', hue = y_test, ax = axes[0], palette = 'husl')
sn.scatterplot(data = X_test,x = 'petal_length', y = 'petal_width', hue = y_predicted_test, ax = axes[1], palette = 'husl')
plt.show()
