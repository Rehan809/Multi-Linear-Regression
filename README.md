# Multi-Linear-Regression
 Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load your dataset
# Replace 'housing_prices.csv' with the path to your dataset
data = pd.read_csv('housing_prices.csv')
# Display the first few rows of the dataset
print(data.head())
# Separate the independent variables (features) and the dependent variable (target)
# Assuming 'Area', 'Floor', 'Room', and 'Code' are column names in the dataset
X = data[['Area', 'Floor', 'Room', 'Code']]
y = data['Price']  # Assuming 'Price' is the column name for the price
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the linear regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict prices for the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
