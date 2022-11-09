import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('/Users/jonathannaumanen/Downloads/data_assignment2.csv')
df = df.dropna(subset=['Land_size'])
df_living_area_and_SellingPrice = df[['Living_area', 'Selling_price']]

xValues = df_living_area_and_SellingPrice['Living_area'].values
yValues = df_living_area_and_SellingPrice['Selling_price'].values

living_area_over_160_and_selling_price_under_3000000 = df_living_area_and_SellingPrice[(df_living_area_and_SellingPrice['Living_area'] > 160) & (df_living_area_and_SellingPrice['Selling_price'] < 3000000)]
print("Living area over 160 and selling price under 3000000 are: \n", living_area_over_160_and_selling_price_under_3000000)

# Plot the data
plt.scatter(xValues, yValues, color='limegreen', linewidths=0.5, edgecolors='black')
plt.xlabel('Living area')
plt.ylabel('Selling price')
plt.title('Living area vs Selling price')
model = LinearRegression().fit(xValues.reshape(-1, 1), yValues.reshape(-1, 1))
interceptValue = model.intercept_
slopeLine = model.coef_
print("Intercept value is: ", interceptValue)
print("Slope line is: ", slopeLine)

y_pred = model.predict(xValues.reshape(-1, 1))
plt.plot(xValues, y_pred, color='red', linewidth=0.5, label='Linear regression line')

# Draw lines from the data points to the regression line
for i in range(len(xValues)):
    plt.plot([xValues[i], xValues[i]], [yValues[i], y_pred[i]], color='black', linestyle='--', linewidth=0.5)

plt.show()




