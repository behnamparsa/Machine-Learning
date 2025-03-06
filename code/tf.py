from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']

(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, train_size=0.7)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(500, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['f1_score'])

model.fit(X_train, y_train, epochs=50)

y_predicted = model.predict(X_test)


print(model.summary())

print("end")