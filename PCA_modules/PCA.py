import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def PCA(raw_data, principal_components):
    # Standardizing
    x_std = StandardScaler().fit_transform(raw_data)       # Produces a 7027x64 matrix (7027 companies, 64 independent variables

    # CALCULATE CORRELATION MATRIX AND ASSOCIATED EIGENVALUES/EIGENVECTORS
    cor_mat = np.corrcoef(x_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]    # Add df.columns[i] after eig_vecs[:,1] as a third column to include variable name

    # SORT
    eig_pairs.sort()
    eig_pairs.reverse()

    # CONSTRUCT PROJECTION MATRIX WITH 7 PRINCIPAL COMPONENTS (~60% INFORMATION RETAINED)
    #pc = principal_components
    matrix_w = np.hstack((eig_pairs[i][1].reshape(64, 1) for i in range(principal_components)))  # Produces a 64 x 'Principal Components' (e.g. 64x7) Matrix

    ''' UN-HIDE CODE TO EXPORT REDUCED DATASET AS A CSV '''
#    matrix_w_df = pd.DataFrame(matrix_w)
#    matrix_w_df.to_csv("matrix_w.csv")

    y = x_std.dot(matrix_w)

    y_df = pd.DataFrame(y)

    return y_df
