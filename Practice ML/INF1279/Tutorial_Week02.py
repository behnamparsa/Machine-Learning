import pandas as pd
import numpy as np
df = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\possum.csv')
#print(df.head())

#-----------------------------------------------------------------

from sklearn.metrics.pairwise import euclidean_distances
#print(type(euclidean_distances(df[['skullw','totlngth']])))

#-------------------------------------------------------------

training = df.iloc[:100]
#print(training)
testing = df.drop(training.index)
features_col = ['hdlngth','skullw','totlngth','taill']
target_col = 'sex'
#print(training[features_col].head())
#print(training[target_col].head())

#______________________________________________________
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training[features_col], training[target_col])

#-----------------------------------------------------

print(testing[features_col])
predicted  = pd.Series((classifier.predict(testing[features_col])))
print(predicted)

#---------------------------------------------------------------
testing['predictionUsingKnn'] = classifier.predict(testing[features_col])
print(testing.head())
#--------------------------------------------------------------

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training[features_col], training[target_col])
print(pd.Series(classifier.predict(testing[features_col])))
prediction = classifier.predict(testing[features_col])
print(prediction)

#-------------------- measuring model's results accuracy --------------------
#confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
conf_matrix = pd.DataFrame(confusion_matrix(testing['sex'], testing['predictionUsingKnn']))
print(conf_matrix)

acc_score = accuracy_score(testing[target_col], prediction)
print(acc_score)

f1_acc_score = f1_score(testing['sex'], testing['predictionUsingKnn'], average='weighted')
print(f1_acc_score)
