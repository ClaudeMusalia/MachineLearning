# importing the dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and analysis
# loading the diabetes dataset to a pandas Dataframe
diabetes_data = pd.read_csv('diabetes.csv')

# attributes of the data set
'''print(diabetes_data.head())
print(diabetes_data.shape)
print(diabetes_data.describe()'''

# outcomes on the dataset
'''print(diabetes_data['Outcome'].value_counts())'''
'''0--> Non-diabetic
1--> diabetic'''

'''print(diabetes_data.groupby('Outcome').mean())'''

# separating the data and labels
X = diabetes_data.drop(columns='Outcome', axis=1)  # axis=1 when dropping a column and axis=0 when dropping a row
Y = diabetes_data['Outcome']

# Data standardization. If there is a difference in the range of values of the different features, the model will
# have a hard time carrying out the prediction, hence standardization.

scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
'''print(standardized_data)'''

X = standardized_data

# Train, test and split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
# stratify means that we make sure that there will be similar proportions of diabetes cases in the training data and
# in the test data.
'''print(X_train.shape, X.shape, X_test.shape)'''  # check the attributes of the original, training and test data.

# Training the model
classifier = svm.SVC(kernel='linear')

# Training the support vector machine Classifier
classifier.fit(X_train, Y_train)

# Model evaluation
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data :', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data :', test_data_accuracy)

# Make a prediction system
input_data = (0, 137, 40, 35, 168, 43.1, 2.288, 33)
# transform the input data into a numpy array using the numpy library
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance. this is because the model was training on the number of
# examples we have and thus will expect the same number of examples. reshaping will tell the model that we are trying
# to predict the label for one instance.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# we need to standardize the input data since we did so earlier for better predictions by the model.
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person in not diabetic')
else:
    print('The person is diabetic')