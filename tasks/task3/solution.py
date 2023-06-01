import numpy as np
from scipy.optimize import approx_fprime

def cel(x):
    # Definicja funkcji celu
    return x[0]**2 + x[1]**2

def metoda_gradientowa(cel, x0, lr, epsilon, max_iter):
    x = x0.copy()
    for i in range(max_iter):
        gradient = approx_fprime(x, cel, epsilon)
        x -= lr * gradient
        if np.linalg.norm(gradient) < epsilon:
            break
    return x

def symulowane_wyzarzanie(cel, x0, T0, alpha, epsilon, max_iter):
    x = x0.copy()
    T = T0
    for i in range(max_iter):
        if T < epsilon:
            break
        x_next = x + np.random.normal(size=x.shape)
        delta_cel = cel(x_next) - cel(x)
        if delta_cel < 0 or np.exp(-delta_cel / T) > np.random.rand():
            x = x_next
        T *= alpha
    return x

def szukanie_przypadkowe(cel, x_min, x_max, max_iter):
    best_x = None
    best_cel = float('inf')
    for i in range(max_iter):
        x = np.random.uniform(x_min, x_max, size=(2,))
        c = cel(x)
        if c < best_cel:
            best_x = x
            best_cel = c
    return best_x

# Przykładowe użycie
x0 = np.array([1.0, 1.0])  # Punkt początkowy
lr = 0.1  # Współczynnik uczenia dla metody gradientowej
T0 = 1.0  # Temperatura początkowa dla symulowanego wyżarzania
alpha = 0.95  # Współczynnik schładzania dla symulowanego wyżarzania
epsilon = 1e-5  # Warunek stopu dla gradientowej i wyżarzania
max_iter = 1000  # Maksymalna liczba iteracji

# Metoda gradientowa
x_opt_gradient = metoda_gradientowa(cel, x0, lr, epsilon, max_iter)
print("Metoda gradientowa:")
print("Minimum znalezione przez metodę gradientową:", x_opt_gradient)
print("Wartość funkcji celu w minimum:", cel(x_opt_gradient))

# Symulowane wyżarzanie
x_opt_sa = symulowane_wyzarzanie(cel, x0, T0, alpha, epsilon, max_iter)
print("Symulowane wyżarzanie:")
print("Minimum znalezione przez symulowane wyżarzanie:", x_opt_sa)
print("Wartość funkcji celu w minimum:", cel(x_opt_sa))

# Szukanie przypadkowe
x_min = -10.0  # Dolne ograniczenie przestrzeni poszukiwań
x_max = 10.0  # Górne ograniczenie przestrzeni poszukiwań
x_opt_random = szukanie_przypadkowe(cel, x_min, x_max, max_iter)
print("Szukanie przypadkowe:")
print("Minimum znalezione przez szukanie przypadkowe:", x_opt_random)
print("Wartość funkcji celu w minimum:", cel(x_opt_random))
