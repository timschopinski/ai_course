
def get_gradient_descent_values(x_test_std, y_test_std, n=1000, factor=0.1):
    theta_0, theta_1 = 0, 0
    for i in range(n):
        theta_0, theta_1 = calculate_gradient_descent(theta_0, theta_1, x_test_std, y_test_std, factor)

    return theta_0, theta_1


def calculate_gradient_descent(theta_0, theta_1, x_data, y_data, factor):
    gradient_m = 0
    gradient_b = 0
    n = len(x_data)

    for i in range(n):
        x = x_data[i]
        y = y_data[i]
        gradient_m += -(2/n) * x * (y - (theta_1 * x + theta_0))
        gradient_b += -(2/n) * (y - (theta_1 * x + theta_0))
    theta_1 = theta_1 - gradient_m * factor
    theta_0 = theta_0 - gradient_b * factor

    return theta_0, theta_1
