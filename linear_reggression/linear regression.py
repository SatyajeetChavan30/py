
# Linear Regression finds the relationship between an independent variable (X) and a dependent variable (Y) using a straight-line equation:

# [Y = wX + b]

# X → Input feature (Years of Experience)
# Y → Target variable (Salary)
# w → Slope or weight
# b → Intercept or bias
# The goal is to find the best values of w and b that minimize prediction error

# Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8')

#"D:\pd\chosen one\ai\linear_reggression\Salary.csv"


#  Load the Dataset

salary_data = pd.read_csv('D:\\pd\chosen one\\ai\\linear_reggression\\Salary.csv')

salary_data.head()


# Understand the Dataset

print('Shape of dataset:', salary_data.shape)
print('\nColumn names:')
print(salary_data.columns)

print('\nData types:')
print(salary_data.dtypes)

print('\nMissing values:')
print(salary_data.isnull().sum())


# Statistical Summary
salary_data.describe()


# Visualize the Relationship Between Variables
plt.figure(figsize=(8,5))
plt.scatter(salary_data.iloc[:,0], salary_data.iloc[:,1], color='darkblue')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')

plt.show()


# Split Features and Target Variable
# X contains the input feature.
# Y contains the output we want to predict.
X = salary_data.iloc[:, :-1].values
Y = salary_data.iloc[:, 1].values

print('X shape:', X.shape)
print('Y shape:', Y.shape)


# Split Data into Training and Testing Sets
# Training data (80%) teaches the model.
# Testing data (20%) evaluates how well the model performs on unseen data.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42
)

print('Training samples:', len(X_train))
print('Testing samples:', len(X_test))


# Train the Linear Regression Model
# The model learns the best-fit line from the training data
model = LinearRegression()

model.fit(X_train, Y_train)


# View the Learned Equation
# After training, the model calculates the slope (weight) and intercept (bias)
weight = model.coef_[0]
bias = model.intercept_

print('Weight (w):', weight)
print('Bias (b):', bias)

print(f'\nEquation: Salary = {weight:.2f} × Experience + {bias:.2f}')


# Predict Salary for Test Data
predictions = model.predict(X_test)

comparison = pd.DataFrame({
    'Experience': X_test.flatten(),
    'Actual Salary': Y_test,
    'Predicted Salary': predictions.round(2)
})

comparison.sort_values('Experience')



# Evaluate Model Performance

# We use multiple evaluation metrics:

# MAE (Mean Absolute Error): Average prediction error.
# MSE (Mean Squared Error): Penalizes larger errors.
# RMSE (Root Mean Squared Error): Error measured in original units.
# R² Score: Measures how well the model explains the data.

# A higher R² score (closer to 1) indicates better performance.
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, predictions)

print(f'MAE  : {mae:.2f}')
print(f'MSE  : {mse:.2f}')
print(f'RMSE : {rmse:.2f}')
print(f'R²   : {r2:.4f}')



# Visualize Actual vs Predicted Values
plt.figure(figsize=(8,5))

plt.scatter(X_test, Y_test, color='red', label='Actual Salary')
plt.plot(X_test, predictions, color='blue', linewidth=2, label='Predicted Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()

plt.show()



# Residual Analysis
# Residuals are the differences between actual and predicted values.

# A good Linear Regression model should have residuals randomly distributed around zero.
residuals = Y_test - predictions

plt.figure(figsize=(8,5))

plt.scatter(predictions, residuals)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel('Predicted Salary')
plt.ylabel('Residual Error')
plt.title('Residual Plot')

plt.show()



# Make a Custom Prediction
experience = np.array([[5]])  # 5 years of experience

predicted_salary = model.predict(experience)

print(f'Predicted salary for {experience[0][0]} years of experience: {predicted_salary[0]:.2f}')