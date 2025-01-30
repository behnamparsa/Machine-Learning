import pandas as pd    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\possum.csv')

training = df.iloc[:20]
testing = df.drop(training.index)
features_col = ['hdlngth','skullw','totlngth','taill']
target_col = 'sex'

# fiting the model: KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training[features_col], training[target_col])


# Model's accuracy on train data
trained = pd.Series(classifier.predict(training[features_col]))
training = pd.DataFrame(df.iloc[:20])
training["KNN_Cluster"] = pd.Series(classifier.predict(training[features_col]))

print(pd.DataFrame(confusion_matrix(training[target_col],trained)))
print(accuracy_score(training[target_col],trained))
#print(training)
