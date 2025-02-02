# Random Forest Molel fitting

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
df = pd.read_pickle('titanic.pickle')
features = df.drop('alive', axis=1)
target = df['alive']


# Convert categorical variables to numerical using one-hot encoding
features_categorical = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(features_categorical, target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=2000, random_state=42)

rf.fit(X_train, y_train)

#Random Forest Model Evaluation
y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.3f}")