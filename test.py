(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, train_size=0.7)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(500, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

model.fit(X_train, y_train, epochs=50)

y_predicted = model.predict(X_test)

print(model.summary())

print("Score:", accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predicted, axis=1)))



