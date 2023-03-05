from typing import Any
import numpy as np
from numpy import ndarray


def calculate_mean_squared_error(theta_0: float, theta_1: float,
                                 x_data: ndarray, y_data: ndarray) -> Any:
    y_pred = theta_0 + theta_1 * x_data
    mse = np.mean((y_data - y_pred) ** 2)
    return mse
