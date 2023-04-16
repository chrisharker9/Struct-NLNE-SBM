# Structure of Nonlinear Node Embeddings in Stochastic Block Models
Here we provide the source code to reproduce the experimental results contained in "Structure of Nonlinear Node Embeddings in Stochastic Block Models."

# Requirements

The necessary requirements can be found [requirments.txt](here). We recommend using `virtualenv` to create a virtual environment, install the necessary requirements, and install the provided `deepwalk` package. 

```
pip install -r requirements.txt 
```

In the project directory containing the `setup.py` file, run

```pip install .```

to install the `deepwalk` package.

# Description of Files

* **deepwalk** : contains code to run the DeepWalk and SGNS models. It also contains code to generate a co-occurrence matrix from a graph. Usage can be found in the class and function docstrings.
    * The `models` sub-package contains code for the deepwalk and negative_sampling classes.
    * The `walks` sub-package contains code to generate a co-occurrence matrix.  
    
* **notebooks** : Contains Jupyter notebooks that reproduce the figures and experimental results in the papaer

    * "Brute Force Optimization" and "Brute Force Optimization - Unequal sized clusters" reproduces the brute force optimization results the plots in Figure 1 in the paper.
    * "Sensitivity to Number of Negative Samples" reproduces the plots shown in Figure 2 of the paper.  