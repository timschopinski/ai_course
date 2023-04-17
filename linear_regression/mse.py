from typing import Any
import numpy as np
from numpy import ndarray


def calculate_mse(
        theta_0: float,
        theta_1: float,
        x_data: ndarray,
        y_data: ndarray
) -> Any:

    y_pred = theta_0 + theta_1 * x_data
    mse = np.mean((y_data - y_pred) ** 2)
    return mse

def calculate_mse_using_loop(
        theta_0: float,
        theta_1: float,
        x_data: ndarray,
        y_data: ndarray
) -> Any:

    n = len(y_data)

    sum_ = 0
    for i in range(n):
        y_pred = theta_0 + theta_1 * x_data[i]
        y = y_data[i]
        result = y - y_pred
        sum_ += result ** 2
    mse = sum_ / n
    return mse
