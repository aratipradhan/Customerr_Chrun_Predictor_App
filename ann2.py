# Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

# Part 1 - Data Preprocessing
dataset = pd.read_csv(r"C:\Users\arati\DATAS SCIENCE NIT\OCTOBER\7.XGBOOST\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values  # Selecting features (ignoring ID and target)
y = dataset.iloc[:, -1].values  # Target variable (Churn or No Churn)

# Encoding categorical data
# Label Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding more hidden layers
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Compiling and Training the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=5)

# Making predictions and evaluating the model
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print("Confusion Matrix:", cm)
print("Accuracy:", ac)

# Saving the model and scaler for the frontend
ann.save('ann_model.h5')
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
