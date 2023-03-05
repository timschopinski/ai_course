import numpy as np


def calculate_mean_squared_error(theta_0, theta_1, x_data, y_data):
    y_pred = theta_0 + theta_1 * x_data
    mse = np.mean((y_data - y_pred) ** 2)
    return mse
