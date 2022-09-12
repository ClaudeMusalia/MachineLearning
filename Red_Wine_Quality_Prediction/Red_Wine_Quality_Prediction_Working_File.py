# Workflow
# 1. Obtain data
# 2. Data analysis
# 3. Data Preprocessing
# 4. Train test split data
# 5. Train/ fit Model
# 6. Evaluate the model

# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data collection
# loading the dataset to a pandas dataframe

wine_data = pd.read_csv('winequality-red.csv')

# check the number of rows and columns
'''wine_data.shape'''

# first 5 rows of the dataset
wine_data.head()

# check for missing values
wine_data.isnull().sum()

# Data analysis and visualization.
# statistical measures of the data set
wine_data.describe()

# number of values for each wine quality
sns.catplot(x='quality', data=wine_data, kind='count')

# check the correlation between the features and the quality
# volatile acidity vs quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_data)

# citric acid vs quality
sns.barplot(x='quality', y='citric acid', data=wine_data)

# find the correlation between the data
correlation = wine_data.corr()

# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

# Data preprocessing
# separate the data and label
X = wine_data.drop('quality', axis=1)

# Label binarization (label encoding), the labels to either good or bad depending on the quality rating.
Y = wine_data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Training and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Training the model
# Random Forest Classifier model. an ensemble model is a model that uses two or more models for prediction.
# A random forest model contains multiple decision tree models, hence and ensemble model.
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Evaluate our model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('The test data accuracy is :', test_data_accuracy)

# Building a predictive system
input_data = (6, 0.32, 0.47, 1.4, 0.055999999999999994, 9.0, 24.0, 0.99695, 3.22, 0.82, 10.3)

# changing the input data to a numpy array. Very useful for processing tuple data. c
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
