import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
print("Program started")
df = pd.read_csv("HousingData.csv")
print("First 5 rows of dataset:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nMissing values :")
print(df.isnull().sum())
df = df.fillna(df.mean())
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
sample_input = X.iloc[[0]]  
predicted_price = model.predict(sample_input)
print("\nSample Prediction:", predicted_price[0])

