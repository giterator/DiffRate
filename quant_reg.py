import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
data = pd.read_csv("jetson_deitS_batch8_etrr_thru.csv") #pd.read_csv("deitS_Rpi_batch1_etrr_thru_gran_1.csv")  #pd.read_csv("jetson_vit_large_patch16_mae_batch8_gran_4_etrr_thru.csv") #  # Replace with your actual CSV file name
vanilla_thru = 56 #5.02 #56
data.columns = ["ETR", "Throughput"]

data['Throughput'] = np.log(data['Throughput'])

# Quantile regression for the median (50th percentile)
mod = smf.quantreg('Throughput ~ ETR', data=data)
res = mod.fit(q=0.001)
# print(res.summary())

coef = res.params
intercept = coef['Intercept']
slope = coef['ETR']

print("Intercept: ", intercept)
print("slope: ", slope)

#example
etrr = 50
ThroughputProxy = np.exp(slope * etrr + intercept)
print(ThroughputProxy)

# # Assuming your CSV has columns 'ETR' and 'Throughput'
# X = data['ETR'].values.reshape(-1, 1)
# y = data['Throughput'].values
# y = np.log(y - vanilla_thru)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Quantile regression for the 50th percentile
# model = QuantileRegressor(quantile=0.5)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# # Extract coefficients and intercept
# coef = model.coef_[0]
# intercept = model.intercept_

# # Construct the equation string
# equation = f"y = {coef:.2f} * X + {intercept:.2f}"
# print(equation)

# #example
# etrr = 50
# ThroughputProxy = np.exp(coef * etrr + intercept)
# print(ThroughputProxy)