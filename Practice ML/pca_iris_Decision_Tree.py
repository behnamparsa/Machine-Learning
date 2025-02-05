from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

iris = load_iris(as_frame = True)

X = iris['data']

y = np.choose(iris['target'], iris['target_names'])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle= True, train_size= 0.7)

pca = PCA(2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_predicted = tree.predict(X_test)

print("accuracy:", accuracy_score(y_test,y_predicted))





df = pd.DataFrame(pca.components_)
df.columns = X.columns

print(df)

fig, axes = plt.subplots(ncols=2)

ax = axes[0]
ax.set_xlabel('Principle Component 1')
ax.set_ylabel('Principle Component 2')
ax.set_title("True")
sn.scatterplot(x = X_test[:,0], y = X_test[:,1], hue = y_test, palette='husl', ax = ax)


ax = axes[1]
ax.set_xlabel('Principle Component 1')
ax.set_ylabel('Principle Component 2')
ax.set_title("Predicted")
sn.scatterplot(x = X_test[:,0], y = X_test[:,1], hue = y_predicted, palette='husl', ax = ax)


plt.show()

