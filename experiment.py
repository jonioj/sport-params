# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Step 1: Load and preprocess your data
# Example: Assuming you have a dataset stored in a CSV file
data = pd.read_csv('data/data.csv')

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['cpet_lens']) # Example: Selecting relevant features
y = data['cpet_lens']

# Optionally, perform data preprocessing steps like encoding categorical variables, handling missing values, etc.

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Step 3: Define the Linear Regression model
reg_model = RandomForestRegressor()

# Step 4: Train the model
reg_model.fit(X_train, y_train)

# Step 5: Evaluate the model
# Predict on the test set
y_pred = reg_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optionally, you can visualize coefficients

reg_model.feature_importances_



# %%

correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Optionally, you can visualize the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title("Feature Correlation Heatmap")
plt.show()

