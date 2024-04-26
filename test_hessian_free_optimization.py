import numpy as np
import pytest

# Import the hessian_free_optimization function from your source code
from basic import hessian_free_optimization

# Define the objective and gradient functions for each test

# 1. Quadratic Function Test
def quadratic_objective(x):
    return x**2

def quadratic_gradient(x):
    return 2 * x

# 2. Rosenbrock Function Test
def rosenbrock_objective(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    df_dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dx1 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

# 3. Rastrigin Function Test
def rastrigin_objective(x):
    return 20 + x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2 * np.pi * x[1])

def rastrigin_gradient(x):
    df_dx0 = 2 * x[0] + 20 * np.pi * np.sin(2 * np.pi * x[0])
    df_dx1 = 2 * x[1] + 20 * np.pi * np.sin(2 * np.pi * x[1])
    return np.array([df_dx0, df_dx1])

def objective_func(x):
    return x[0]**2 + 2 * x[1]**2

def gradient_func(x):
    return np.array([2 * x[0], 4 * x[1]])




# Define test functions for each test case

def test_quadratic_function():
    x_min = hessian_free_optimization(quadratic_objective, quadratic_gradient, np.array([2.0]))
    assert np.isclose(x_min, 0.0)

def test_rosenbrock_function():
    x_min = hessian_free_optimization(rosenbrock_objective, rosenbrock_gradient, np.array([-2.0, 2.0]))
    assert np.allclose(x_min, [1.0, 1.0])

def test_rastrigin_function():
    x_min = hessian_free_optimization(rastrigin_objective, rastrigin_gradient, np.array([2.0, -2.0]))
    assert np.allclose(x_min, [0.0, 0.0])

def test_func():
    x_min = hessian_free_optimization(objective_func, gradient_func, np.array([2.0, 1.0]))
    assert np.allclose(x_min, [2.0, 1.0])
