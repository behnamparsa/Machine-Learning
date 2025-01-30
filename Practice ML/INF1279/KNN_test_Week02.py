import pandas as pd    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\possum.csv')

training = pd.DataFrame(df.iloc[:20])
testing = df.drop(training.index)
features_col = ['hdlngth','skullw','totlngth','taill']
target_col = 'sex'

# fiting the model: KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training[features_col], training[target_col])


# Model's accuracy on train data
training["KNN_Cluster"] = classifier.predict(training[features_col])
#training["KNN_Cluster"] = trained
#-------------------------------------
print("KNN Model accuracy on train data: Conf. Matrix, Accuracy Score and F1_Score")
print(pd.DataFrame(confusion_matrix(training[target_col],training["KNN_Cluster"])))
print(accuracy_score(training[target_col],training["KNN_Cluster"]))
print(f1_score(training[target_col],training["KNN_Cluster"], average= 'weighted'))

# Model's accuracy on test data
testing['predictionUsingKnn'] = classifier.predict(testing[features_col])
#---------------------------------------------------------------
print("KNN Model accuracy on test data: Conf. Matrix, Accuracy Score and F1_Score")
print(pd.DataFrame(confusion_matrix(testing[target_col], testing['predictionUsingKnn'])))
print(accuracy_score(testing[target_col],testing['predictionUsingKnn']))
print(f1_score(testing[target_col],testing['predictionUsingKnn'],average = 'weighted'))


