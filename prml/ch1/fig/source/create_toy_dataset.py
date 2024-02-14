import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from jax import Array
from typing import Tuple


def create_toy_dataset(key: ArrayLike, sample_size: int, std: ArrayLike) -> Tuple[Array, Array]:
    """Create dataset with noise added to sin(2πx)
    
    Args:
        key: random key
        sample_size: number of samples
        std: standard deviation of noise
        
    Returns:
        x: input data
        t: output data"""

    x = jnp.linspace(0, 1, sample_size)
    t = jnp.sin(2 * jnp.pi * x) + std * jax.random.normal(key, (sample_size,))
    return x, t

def save(x, t):
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    data = pd.DataFrame({'x': x, 't': t})
    data.to_csv(path, index=False)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    key = jax.random.PRNGKey(0)
    sample_size = 10
    std = 0.3
    x, t = create_toy_dataset(key, sample_size, std)
    print(x, t)
    # Save the dataset
    #save(x, t)
    
    ax = plt.axes(xlim=(-0.01, 1.01), ylim=(-1.5, 1.5))
    ax.scatter(x, t, s=20, facecolor="red")
    
    plot_x = jnp.linspace(0, 1.0, 100)
    ax.plot(plot_x, jnp.sin(2 * jnp.pi * plot_x), color="green", label="sin(2πx)")
    
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    
    plt.show()