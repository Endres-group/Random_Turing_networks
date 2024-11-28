import numpy as np

# Original Function: Gaussian matrices, neglect self-interactions, with fixed degradation
def gen_matrix_F(n, vr):
    """
    Generate a Gaussian matrix neglecting self-interactions (diagonal = 0),
    with fixed self-degradation (-1 on diagonal).
    """
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)  # Remove self-interactions
    A = -I + B  # Add fixed self-degradation (-1 on diagonal)
    D = np.zeros((n, n))
    return A, D

# Function 1: Full Gaussian matrices with self-interactions and fixed self-degradation
def gen_matrix_F_full_Gaussian_fixed_degradation(n, vr):
    """
    Generate a full Gaussian matrix with self-interactions 
    and fixed self-degradation (-1 on diagonal).
    """
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)  # All elements drawn from Gaussian
    A = -np.eye(n) + B  # Add fixed self-degradation (-1 on diagonal)
    D = np.zeros((n, n))
    return A, D

# Function 2: Full Gaussian matrices with self-interactions but no fixed self-degradation
def gen_matrix_F_full_Gaussian_no_degradation(n, vr):
    """
    Generate a full Gaussian matrix with self-interactions 
    but without fixed self-degradation.
    """
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)  # All elements drawn from Gaussian
    A = B  # No fixed self-degradation added
    D = np.zeros((n, n))
    return A, D

# Function 3: Anti-symmetric matrices with fixed self-degradation and no self-interactions
def gen_matrix_F_antisymmetric(n, vr):
    """
    Generate an anti-symmetric matrix with fixed self-degradation (-1 on diagonal)
    and no self-interactions.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            random_val = np.sqrt(vr) * np.random.randn()
            A[i, j] = random_val
            A[j, i] = -random_val  # Anti-symmetric property
    np.fill_diagonal(A, -1)  # Set diagonal elements to -1 (fixed self-degradation)
    D = np.zeros((n, n))
    return A, D
