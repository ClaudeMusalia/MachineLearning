import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading the data from a csv file and turning it into a pandas Dataframe
gold_data = pd.read_csv('gld_price_data.csv')

'''# Print the first five rows
print(gold_data.head())

# Print the last five rows
print(gold_data.tail())

# number of rows and columns
print(gold_data.shape)

# check for missing values
print(gold_data.isnull().sum())

# getting statistical measures of the data
print(gold_data.describe())'''

# checking of correlation between different features of the dataset. Positive and negative correlation
correlation = gold_data.corr()

# construct a heatmap to understand the correlation between different features of the dataset
'''plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()'''

'''# from the data, we find that gold is positively correlated to silver, and slightly negatively correlated to USD
print(correlation['GLD'])

# checking the distribution of the GLD Price
sns.displot(gold_data['GLD'], color='green')
plt.show(block=True)'''

# splitting the features and Target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# splitting data to testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# training the model: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# testing the accuracy, using the R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print('R squared error:', error_score)

# comparing the actual values and predicted values in a plot
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price Vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show(block=True)