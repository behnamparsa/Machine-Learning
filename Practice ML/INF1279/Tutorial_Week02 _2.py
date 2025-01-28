from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#-------------- KNN supervised -------------------------------------------
def KNeighbors_Classifier(X_train, y_train, X_test):

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predicted = pd.Series((classifier.predict(X_test)))
    X_test['predictionUsingKnn'] = classifier.predict(X_test)

    return X_test, predicted


#--------------Kmeans--------------------------------
def cluster(df, n):
    #X = df[['hdlngth','skullw']]
    classifier = KMeans(n_clusters= n)
    classifier.fit(df)
    predicted_KM = pd.Series((classifier.predict(df)))
    df['cluster'] = classifier.labels_


#---------- plot for KMeans ---------------------------
def plot(df):
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot()
    ax.scatter(df['hdlngth'], df['skullw'], c=df['cluster'], cmap='plasma')
    ax.set_xlabel('hdlngth')
    ax.set_ylabel('skullw')

    #df.drop(['sex','Pop'], axis=1, inplace=True)
    
    means = df.groupby(by='cluster').mean()

    ax.scatter(means['hdlngth'], means['skullw'], color='red', s=100)

    plt.show()

def main():

    df = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\possum.csv')
    #print(df.head())
    #print(type(euclidean_distances(df[['skullw','totlngth']])))
#---------------------- data prep ------------------------------------
    training = df.iloc[:20]
    testing = df.drop(training.index)
    #testing = testing_0.iloc[:5]
   
    features_col = ['hdlngth','skullw','totlngth','taill']
    target_col = 'sex'

    testing_KNN, predicted = KNeighbors_Classifier(training[features_col],training[target_col],testing[features_col])
#-------------------------testing KNN -----------------------------------------------
    
    conf_matrix = pd.DataFrame(confusion_matrix(testing[target_col], testing_KNN['predictionUsingKnn']))
    print(conf_matrix)
    acc_score = accuracy_score(testing[target_col], predicted)
    print(acc_score)
    f1_acc_score = f1_score(testing[target_col], testing_KNN['predictionUsingKnn'], average='weighted')
    print(f1_acc_score)
#------------------------- KMeans -----------------------------------------------
#is not useful for this dataset
    training_KM = df.iloc[:,:]
    #testing_KM = df.drop(training_KM.index)
           
    features_col = ['hdlngth','skullw','totlngth','taill']
    target_col = 'sex'    
    df = training_KM[features_col]
    cluster(df, 2)
    plot(df)
    
main()
