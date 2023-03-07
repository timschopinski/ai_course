import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from data import get_data, inspect_data, split_data
from lab1.closed_form_solution import get_closed_form_solution
from lab1.gradient import get_gradient_descent_values
from lab1.mse import calculate_mse, calculate_mse_using_loop

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_0, theta_1 = get_closed_form_solution(x_train, y_train)


# TODO: calculate error
mse = calculate_mse_using_loop(theta_0, theta_1, x_test, y_test)
print(f'MSE with for loop: {mse}')

mse = calculate_mse(theta_0, theta_1, x_test, y_test)
print(f'MSE: {mse}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = theta_0 + theta_1 * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: standardization
x_train_std: ndarray = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_std: ndarray = (y_train - np.mean(y_train)) / np.std(y_train)

x_test_std: ndarray = (x_test - np.mean(x_train)) / np.std(x_train)
y_test_std: ndarray = (y_test - np.mean(y_train)) / np.std(y_train)


# TODO: calculate theta using Batch Gradient Descent
theta_0, theta_1 = get_gradient_descent_values(x_test_std, y_test_std, factor=0.001)


# TODO: calculate error
mse = calculate_mse(theta_0, theta_1, x_test_std, y_test_std)
print(f'MSE: {mse}')

# plot the regression line
x = np.linspace(min(x_test_std), max(x_test_std), 100)
y = theta_0 + theta_1 * x
plt.plot(x, y)
plt.scatter(x_test_std, y_test_std)
plt.xlabel('Weight (standardized)')
plt.ylabel('MPG (standardized)')
plt.show()

theta_0, theta_1 = get_closed_form_solution(x_test_std, y_test_std)
mse = calculate_mse(theta_0, theta_1, x_test_std, y_test_std)
print(f'MSE: {mse}')

# plot the regression line
x = np.linspace(min(x_test_std), max(x_test_std), 100)
y = theta_0 + theta_1 * x
plt.plot(x, y)
plt.scatter(x_test_std, y_test_std)
plt.xlabel('Weight (standardized) CFS')
plt.ylabel('MPG (standardized) CFS')
plt.show()
