# importing the dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and processing
# loading the csv data into a pandas dataFrame

heart_data = pd.read_csv('heart_disease_data.csv')

''' # print first 5 rows of data set
print(heart_data.head())

# print last 5 rows of data set
print(heart_data.head()

# print various statistical measures of dataset
print(heart_data.describe()

# print number of rows and columns in dataset
print(heart_data.shape)

# getting some information about the dataset
print(heart_data.info())

# checking for missing values
print(heart_data.isnull().sum())'''

# checking the distribution of the target variable
heart_data['target'].value_counts()

# 1 represents  a defective heart while  0 represents a healthy heart

# splitting the features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Training the model
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# training the logistic model with the training data
model.fit(X_train, Y_train)

# model evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ', test_data_accuracy)

# Building a predictive system
input_data = (68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3)

# change the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have heart disease')
else:
    print('The person has heart disease')
