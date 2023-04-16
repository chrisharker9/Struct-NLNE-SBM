import numpy as np

def construct_co_occurrence_matrix(A, n_walks, walk_length, window_size):
    """
    ====================================================
    Construct a co-occurrence matrix using random walks.
    ====================================================

    Parameters
    ----------
    A: Numpy array of shape (n, n)
      Adjacency Matrix
    n_walks: int
      Number of random walks starting at each node.
    walk_length: int
      Length of the random walk.
    window_size: int
      Size of the window.

    Returns
    -------
    C: Numpy Array of shape (n, n)
        Co-occurrence matrix
    """
    
    n_nodes = A.shape[0]

    # Initialize empty array
    C = np.zeros(shape=(n_nodes, n_nodes))

    for v in range(n_nodes):
        for nw in range(n_walks):
          walk = [v]
          for l in range(walk_length):    
              neighbors = np.where(A[walk[l], :] == 1)[0]
              walk += [np.random.choice(neighbors)]
          
          for i in range(len(walk)-1):
              w = walk[i]
              for k in range(1, window_size+1):
                  if i+k < len(walk):
                    C[w, walk[i+k]] += 1
                    C[walk[i+k], w] += 1
    return C

    