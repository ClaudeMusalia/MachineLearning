# importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# importing the Boston House Price data
house_price_dataset = sklearn.datasets.load_boston()
'''print(house_price_dataset)'''
# Loading dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# add the target (house prices) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target

house_price_dataframe.head()

# checking the number of rows and columns in the data frame
'''house_price_dataframe.shape

# checking for missing values
house_price_dataframe.isnull().sum()

# statistical measures of the dataset
house_price_dataframe.describe()'''

# understanding the correlation between various features
# If the variables/ features are positively correlated or vice versa
correlation = house_price_dataframe.corr()  # this will get the correlation between all the features/variables.

# constructing a heatmap to understand the correlation. used to understand the data more better
'''plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()'''

# splitting the data and target (Price)
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

# splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.2, random_state=1)

# train our model
# we are using XGBoost Regressor algorithm. A type of decision tree algorithm.

# loading the model
model = XGBRegressor()

# fitting the model
model.fit(X_train, Y_train)

# evaluate our model
# get the accuracy on prediction on training data
training_data_prediction = model.predict(X_train)

# r squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print("Mean absolute error : ", score_2)

# prediction on test data
test_data_prediction = model.predict(X_test)

# r squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# mean absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

# visualize the actual prices and predicted prices
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price Vs Predicted Price")
plt.show()