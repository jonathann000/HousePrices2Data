import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('data_assignment2.csv')
df = df.dropna(subset=['Land_size'])
df_living_area_and_SellingPrice = df[['Living_area', 'Selling_price']]

xValues = df_living_area_and_SellingPrice['Living_area'].values
yValues = df_living_area_and_SellingPrice['Selling_price'].values

living_area_over_160_and_selling_price_under_3000000 = df_living_area_and_SellingPrice[
    (df_living_area_and_SellingPrice['Living_area'] > 160) & (
            df_living_area_and_SellingPrice['Selling_price'] < 3000000)]
print("Living area over 160 and selling price under 3000000 are: \n",
      living_area_over_160_and_selling_price_under_3000000)

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

# Print the predicted selling price for houses with 100, 150 and 200 square meters
print("Predicted selling price for houses with 100, 150 and 200 square meters are: \n",
      model.predict([[100], [150], [200]]))

# Draw lines from the data points to the regression line
for i in range(len(xValues)):
    plt.plot([xValues[i], xValues[i]], [yValues[i], y_pred[i]], color='black', linestyle='--', linewidth=0.5)

# Summarize the model with statsmodels
X = sm.add_constant(xValues)
model = sm.OLS(yValues, X)
results = model.fit()
print(results.summary())

plt.show()



from sklearn.datasets import load_iris

# Plot Confusion Matrix with iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression

# Increase max iterations to avoid convergence warning
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
logreg.fit(X_train, y_train)
# Predict for One Observation (image)
logreg.predict(X_test[0].reshape(1, -1))
# Predict for Multiple Observations (images) at Once
logreg.predict(X_test[0:10])
y_pred = logreg.predict(X_test)
score = logreg.score(X_test, y_test)

from sklearn import metrics

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.savefig('Confusion-Matrix.png')

# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

# Different values of k
k_range = range(1, 102, 10)

# Loop through different values of k, with uniform and distance weights
# and plot a confusion matrix for each
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.savefig('Confusion-Matrix-knn-k-' + str(k) + '.png')
    print("Accuracy is ", metrics.accuracy_score(y_test, y_pred), "for K-Value:", k, "and uniform weights")

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.savefig('Confusion-Matrix-knn-k-' + str(k) + '-distance.png')
    print("Accuracy is ", metrics.accuracy_score(y_test, y_pred), "for K-Value:", k, "and distance weights")

plt.show()
