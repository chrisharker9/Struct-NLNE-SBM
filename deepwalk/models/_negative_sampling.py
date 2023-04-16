import numpy as np
from scipy.special import expit


def negative_sampling(C, d=2, n_iter=100, eta=0.1, b=2, seed=1234):
    """
    ========================================
    DeepWalk/Node2Vec with Negative Sampling
    ========================================

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
    b: int
       Number of negative samples
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

    Ct = C.sum()
    Ci = np.diag(C.sum(axis=1))
    Cj = np.diag(C.sum(axis=0))
    for t in range(n_iter):
        Q = expit(X @ Y.T)
        gradx = (C * (1-Q) - (b/Ct)*(Ci @ (Q @ Cj))) @ Y
        grady = (C * (1-Q.T) - (b/Ct)*(Ci @ (Q.T @ Cj))) @ X
        X = X + eta*gradx
        Y = Y + eta*grady
        
    return X, Y