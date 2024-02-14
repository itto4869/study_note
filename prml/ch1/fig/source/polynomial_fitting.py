import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

@partial(jax.jit, static_argnames=("M",))
def polynomial(x: Array, w: Array, M: int) -> Array:
    """Calculate the output of a polynomial model given input data and weights.

    Args:
        x: Input data, a 1D array.
        w: Weights for the polynomial model, a 1D array where w[i] is the weight for x^i.
        M: Degree of the polynomial model.

    Returns:
        The output data as a 1D array, calculated as the dot product of the Vandermonde matrix of x and the weights w.
    """
    
    xs = jnp.vander(x, M + 1, increasing=True)
    return jnp.dot(xs, w)

@jax.jit
def sum_of_squares_error(y: Array, t: Array) -> Array:
    """Calculate the sum of squares error of the predicted output y and the target output t.

    Args:
        y: Predicted output, a 1D array.
        t: Target output, a 1D array.

    Returns:
        The sum of squares error as a scalar.
    """
    return 0.5 * jnp.sum((y - t)**2)

@partial(jax.jit, static_argnames=("M",))
def solve_error_function(x: Array, t: Array, M: int) -> Array:
    """Solve the polynomial curve fitting problem.

    Args:
        x: Input data, a 1D array.
        t: Target output, a 1D array.
        M: Degree of the polynomial model.

    Returns:
        The weights for the polynomial model, a 1D array.
    """
    xs = jnp.vander(x, M + 1, increasing=True)
    return jnp.linalg.solve(jnp.dot(xs.T, xs), jnp.dot(xs.T, t))

def save_weight(ws):
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(__file__), 'weights.csv')
    Ms = list(ws.keys())
    
    data = pd.concat({f'M={M}': pd.DataFrame(ws[M]) for M in Ms}, axis=1)
    data.to_csv(path, index=False)

def save_dataset(x, t):
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/toy_data.txt')
    data = pd.DataFrame({'x': x, 't': t})
    data.to_csv(path, index=False, sep=" ")

def save_fit_curve(ys, x):
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/plot_data_poly_fit.txt')
    Ms = list(ys.keys())
    
    data = pd.concat({f'M={M}': pd.DataFrame(ys[M]) for M in Ms}, axis=1)
    data["x"] = x
    data.to_csv(path, index=False, sep=" ")

if __name__ == "__main__":
    from create_toy_dataset import create_toy_dataset
    jax.config.update("jax_enable_x64", True)
    
    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key)
    sample_size = 10
    std = 0.3
    save = True
    Ms = [0, 1, 3, 9]
    ws = {}
    x, t = create_toy_dataset(key, sample_size, std)
    for M in Ms:
        w = solve_error_function(x, t, M)
        ws[M] = w

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(int(len(Ms)/2), len(Ms)-int(len(Ms)/2), figsize=(16, 9))
    axes = axes.flatten()

    def plot(ax, x, t, w, M):
        ax.scatter(x, t, s=20, facecolor="red")
        plot_x = jnp.linspace(0, 1.0, 100)
        ax.plot(plot_x, jnp.sin(2 * jnp.pi * plot_x), color="green", label="sin(2πx)")
        ax.plot(plot_x, polynomial(plot_x, w, M), color="blue", label="polynomial")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"M={M}")
    
    for ax, M in zip(axes, Ms):
        plot(ax, x, t, ws[M], M)

    plt.tight_layout()
    
    if save:
        #save_weight(ws)
        save_dataset(x, t)
        plt.savefig("polynomial_fitting.png")
        
        plot_x = jnp.linspace(0, 1.0, 100)
        ys = {}
        for M in Ms:
            y = polynomial(plot_x, ws[M], M)
            ys[M] = y
        save_fit_curve(ys, plot_x)
    
    plt.show()