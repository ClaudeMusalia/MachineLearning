# Problem statement

# predict the prices of used cars using various information about the vehicles i.e. car brand, year, sold price,
# present price etc.
# Using the linear regression model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Data collection and processing
# loading the data from the csv file to pandas dataFrame
car_dataset = pd.read_csv('car data.csv')

# inspecting the first five rows of the dataFrame
car_dataset.head()

# checking the number of rows and columns
Rows_and_columns = car_dataset.shape

# getting more information in the dataset
'''more_info = car_dataset.info()'''

# checking the number of missing values
car_dataset.isnull().sum()

# checking the distribution of categorical data
'''print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())'''

# encoding the categorical data. change text values to numerical values for the model to read.
# encoding the fuel type column
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# encoding the Seller type column
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# encoding the transmission column
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# check if encoding has been done properly.
'''print(car_dataset.head())'''

X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# splitting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Training the model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Evaluate the normal
# prediction on training data
training_data_prediction = lin_reg_model.predict(X_train)

# R squared error
error_score_1 = metrics.r2_score(Y_train, training_data_prediction)
print("R squared error : ", error_score_1)

# visualize the actual price and predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# prediction of test data
test_data_prediction = lin_reg_model.predict(X_test)

# R squared error
error_score_2 = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score_2)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
