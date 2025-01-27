"""
Load the iris data.

Fit a k-means model to the data (3 clusters)

use the predict method of the model to predict the culsters

plot the predicted clustered side-by-side with the actual culsters

can you use an r squared score to compare the predicted with the acutal clusters?
If not, why not?

"""


import sklearn.datasets as ds
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from itertools import permutations
import numpy as np
from sklearn.metrics import normalized_mutual_info_score



data = ds.load_iris(as_frame = True)


def find_best_perm(y_true,y_perdicted):
    scores = {}
    for perm in permutations(range(max(y_true)+1)):
        score = r2_score(y_true,np.choose(y_perdicted, perm))
        scores[score] = perm
    
    max_score = max(scores.keys())
    return scores[max_score]

def main():

    target = data['target']
    target_names= data['target_names']

    df = data['data']
    df['species_index'] = target
    df['species'] = target.map(lambda n: target_names[n])

    X = df.iloc[:,0:4]
    X = StandardScaler().fit_transform(X)
    # ----------- Model fitting ------------------------
    model = KMeans(n_clusters= 3)               
    model.fit(X)

    predicted_target = model.predict(X)
    # ----------------------------------

    # ----------- prep to graph the results ------------------------
    x_label = df.columns[2]
    y_label = df.columns[3]
    x = df[x_label]
    y = df[y_label]
# ----------------- we move this up to plot the actual and predicted clusters with the same color
    perm = find_best_perm(target, predicted_target) # call this funtion to measure the clustering accuracy with r2 Score
    predicted_target = np.choose(predicted_target, perm)
#--------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize = (16,9))

    ax = fig.add_subplot(121)
    ax.set_title("True Clusters")           # graph the Actul data
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y , c = target, cmap = "plasma")


    ax = fig.add_subplot(122)
    ax.set_title("Predicted Clusters")      # graph the prediction
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y , c = predicted_target, cmap = "plasma")

    plt.show()
    # ----------- compare the Actual data vs prediction ------------------------
    pd.set_option("display.max_rows", 150)
    df_clusters = pd.DataFrame()
    df_clusters['True'] = target
    df_clusters['Predicted'] = predicted_target

    print(df_clusters) 
    # ----------- accuracy check with r2 score ------------------------
    #perm = find_best_perm(target, predicted_target) # call this funtion to measure the clustering accuracy with r2 Score # moved up to before graphs(same cl color)
    print(r2_score(target, np.choose(predicted_target, perm)))  # call this funtion to measure the clustering accuracy with r2 Score

    print(normalized_mutual_info_score(target, predicted_target))
    




main()
