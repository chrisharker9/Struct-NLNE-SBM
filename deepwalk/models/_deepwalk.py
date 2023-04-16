import numpy as np
from scipy.special import softmax


def deepwalk(C, d=2, n_iter=100, eta=0.1, seed=1234):
    """
    Deepwalk embeddings using Vanilla Skip-gram.

    Parameters
    ----------
    C: Numpy Array of shape (n, n)
        Co-occurrence matrix
    d: int
       Dimension of node and context embeddings
    n_iter: int
       Number of iterations
    eta: float
       Step-size
    seed: int
       Seed for Random Number Generator

    Returns
    -------
    X: Numpy Array of shape (n, d)
        Node embeddings. The embedding for node
        i is accessed via X[i, :]
    Y: Numpy Array of shape (n, d)
        Context embeddings. The embedding for node
        i is accessed via Y[i, :]

    """
    
    
    n = C.shape[0]

    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(n, d))
    np.random.seed(seed+1)
    Y = np.random.uniform(-1, 1, size=(n, d))

    one = np.ones((n, 1))
    temp = C @ one
    D = np.diag( (C @ one)[:, 0] )

    X_grad_list = []
    Y_grad_list = []
    loss = [(C * softmax(X @ Y.T, axis=1)).sum()]

    for t in range(n_iter):
        Q = softmax(X @ Y.T, axis=1)
        X_grad = ((D @ Q) - C) @ Y
        Y_grad = ( ((Q @ D) - C).T ) @ X
        X = X - eta*X_grad
        Y = Y - eta*Y_grad

    return X, Y