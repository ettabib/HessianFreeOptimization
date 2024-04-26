import numpy as np

def hessian_free_optimization(objective_func, gradient_func, x0, max_iter=100, tol=1e-6, damping=1e-4, verbose=True):
    x = x0.copy()
    for i in range(max_iter):
        # Compute the gradient
        gradient = gradient_func(x)
        
        # Compute the Hessian-vector product using finite differences
        def hessian_vector_product(v):
            eps = 1e-4
            return (gradient_func(x + eps * v) - gradient) / eps
        
        # Compute the search direction using conjugate gradient method
        def conjugate_gradient(b, max_cg_iter=100, cg_tol=1e-6):
            p = b.copy()
            r = b.copy()
            x = np.zeros_like(b)
            for j in range(max_cg_iter):
                Ap = hessian_vector_product(p)
                alpha = np.dot(r, r) / (np.dot(p, Ap) + damping * np.dot(p, p))
                x += alpha * p
                r_new = r - alpha * Ap
                beta = np.dot(r_new, r_new) / np.dot(r, r)
                p = r_new + beta * p
                r = r_new
                if np.linalg.norm(r) < cg_tol:
                    break
            return x
        
        # Compute the search direction
        search_direction = conjugate_gradient(-gradient)
        
        # Perform a line search to determine the step size
        step_size = 1.0
        while objective_func(x + step_size * search_direction) > objective_func(x) + 1e-4 * step_size * np.dot(gradient, search_direction):
            step_size *= 0.5
        
        # Update the parameters
        x += step_size * search_direction
        
        # Print verbose information
        if verbose:
            print(f"Iteration {i+1}")
            print(f"  Current point: {x}")
            print(f"  Objective function value: {objective_func(x)}")
            print(f"  Gradient norm: {np.linalg.norm(gradient)}")
            print(f"  Step size: {step_size}")
            print()
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            if verbose:
                print("Convergence reached!")
            break
    
    if verbose:
        print("Final results:")
        print(f"  Optimal point: {x}")
        print(f"  Objective function value: {objective_func(x)}")
    
    return x

# Define the objective function and its gradient
def objective_func(x):
    return x[0]**2 + 2 * x[1]**2

def gradient_func(x):
    return np.array([2 * x[0], 4 * x[1]])

# Set the initial point
x0 = np.array([2.0, 1.0])

# # Run the Hessian-free optimization algorithm with verbose output
# optimal_point = hessian_free_optimization(objective_func, gradient_func, x0, verbose=True)

# # Define the objective and gradient functions for each test

# # 1. Quadratic Function Test
# def quadratic_objective(x):
#     return x**2

# def quadratic_gradient(x):
#     return 2 * x

# # 2. Rosenbrock Function Test
# def rosenbrock_objective(x):
#     return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# def rosenbrock_gradient(x):
#     df_dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
#     df_dx1 = 200 * (x[1] - x[0]**2)
#     return np.array([df_dx0, df_dx1])

# # 3. Rastrigin Function Test
# def rastrigin_objective(x):
#     return 20 + x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2 * np.pi * x[1])

# def rastrigin_gradient(x):
#     df_dx0 = 2 * x[0] + 20 * np.pi * np.sin(2 * np.pi * x[0])
#     df_dx1 = 2 * x[1] + 20 * np.pi * np.sin(2 * np.pi * x[1])
#     return np.array([df_dx0, df_dx1])

# # Run the tests

# # 1. Quadratic Function Test
# x_min = hessian_free_optimization(quadratic_objective, quadratic_gradient, np.array([2.0]))
# print("Quadratic Function Minimum:", x_min)

# # 2. Rosenbrock Function Test
# x_min = hessian_free_optimization(rosenbrock_objective, rosenbrock_gradient, np.array([-2.0, 2.0]))
# print("Rosenbrock Function Minimum:", x_min)

# # 3. Rastrigin Function Test
# x_min = hessian_free_optimization(rastrigin_objective, rastrigin_gradient, np.array([2.0, -2.0]))
# print("Rastrigin Function Minimum:", x_min)