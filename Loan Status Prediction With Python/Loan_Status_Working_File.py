# importing the dependencies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and processing
# loading the dataset to pandas dataFrame
loan_dataset = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

# check various metrics of the data
'''print(loan_dataset.shape)
print(loan_dataset.head())
print(loan_dataset.describe()'''

# number of missing values in each column
loan_dataset.isnull().sum()

# dropping missing values, drop na means the missing values will be dropped
loan_dataset = loan_dataset.dropna()

# number of missing values in each column
loan_dataset.isnull().sum()

# label encoding. This changes the labels/ values in a given column to numerical values for easy processing. Uses a
# dictionary format.
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# printing the first 5 rows of the dataFrame
loan_dataset.head()

# Dependent column values
loan_dataset['Dependents'].value_counts()

# replacing the value 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# dependent values
loan_dataset['Dependents'].value_counts()

# Data Visualization. Important for analysis, to find the relationship between different features.
# education and loan status
'''sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.show()

# marital status & loan status
sns.countplot(x='Married', hue='loan_Status', data=loan_dataset)
plt.show()'''

# convert categorical columns to numerical values
loan_dataset.replace(
    {'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
     'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, "Not Graduate": 0}},
    inplace=True)

# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Training the model
# Support Vector Machine Model. SVC means support vector classifier for the classifier problem.
classifier = svm.SVC(kernel='linear')

# Training the support Vector Machine Model
classifier.fit(X_train, Y_train)

# Model Evaluation
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)

