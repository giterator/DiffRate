import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("etrr_thru.csv") #pd.read_csv("jetson_vit_large_patch16_mae_batch8_gran_4_etrr_thru.csv") #  # Replace with your actual CSV file name
vanilla_thru = 0 #5.02 #56
data.columns = ["ETR", "Throughput"]

# Assuming your CSV has columns 'ETR' and 'Throughput'
X = data['ETR'].values.reshape(-1, 1)
y = data['Throughput'].values
y = np.log(y - vanilla_thru)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Regression
degree = 1  # We'll try 2nd, 3rd, and 4th degree polynomials

poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_model = LinearRegression() #fit_intercept=False
poly_model.fit(X_poly_train, y_train)

y_pred_poly = poly_model.predict(X_poly_test)
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_poly):.4f}")

# Get the coefficients
coefficients = poly_model.coef_
intercept = poly_model.intercept_

# Print the resulting polynomial regression expression
features = poly_features.get_feature_names_out()
print(f"Intercept: {intercept}")
for coef, feature in zip(coefficients, features):
    print(f"{feature}: {coef}")

#example
etrr = 50
ThroughputProxy = np.exp(coefficients[1] * etrr + intercept) #coefficients[2] * etrr**2 + coefficients[1] * etrr + intercept
print(ThroughputProxy)