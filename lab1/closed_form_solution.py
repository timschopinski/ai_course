import numpy as np


def get_closed_form_solution(x_data, y_data):
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    xy_mean = np.mean(x_data * y_data)
    x_squared_mean = np.mean(x_data ** 2)
    theta_1 = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    theta_0 = y_mean - theta_1 * x_mean
    return theta_0, theta_1
