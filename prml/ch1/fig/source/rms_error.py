import jax
import jax.numpy as jnp
from jax import Array
from polynomial_fitting import polynomial, solve_error_function
from create_toy_dataset import create_toy_dataset

@jax.jit
def rms_error(y: Array, t: Array) -> Array:
    """Calculate the root mean square error of the predicted output y and the target output t.

    Args:
        y: Predicted output, a 1D array.
        t: Target output, a 1D array.

    Returns:
        The root mean square error as a scalar.
    """
    return jnp.sqrt(jnp.mean((y - t)**2))

def save_error(train_errors, test_errors):
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/rms_error.txt')
    data = pd.DataFrame({'M': list(train_errors.keys()), 'train_error': list(train_errors.values()), 'test_error': list(test_errors.values())})
    data.to_csv(path, index=False, sep=" ")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)
    train_key, test_key = jax.random.split(key)
    train_size = 10
    test_size = 100
    std = 0.3
    
    x_train, t_train = create_toy_dataset(train_key, train_size, std)
    x_test, t_test = create_toy_dataset(test_key, test_size, std)
    
    Ms = range(10)
    ws = {}
    train_errors = {}
    test_errors = {}
    
    for M in Ms:
        w = solve_error_function(x_train, t_train, M)
        ws[M] = w
        y_train = polynomial(x_train, w, M)
        y_test = polynomial(x_test, w, M)
        train_errors[M] = rms_error(y_train, t_train)
        test_errors[M] = rms_error(y_test, t_test)
    
    save_error(train_errors, test_errors)