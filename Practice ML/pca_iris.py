from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd

iris = load_iris(as_frame = True)

X = iris['data']

y = np.choose(iris['target'], iris['target_names'])

print(X)
print(y)

pca = PCA(2)

components = pca.fit_transform(X)


print(pca.explained_variance_ratio_)
df = pd.DataFrame(pca.components_)
df.columns = X.columns

print(df)

fig, ax = plt.subplots()
ax.set_xlabel('Principle Component 1')
ax.set_ylabel('Principle Component 2')
sn.scatterplot(x =components[:, 0], y = components[:, 1], hue = y, palette= 'husl', ax = ax)

plt.show()

