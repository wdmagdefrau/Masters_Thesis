import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PredictBankruptcy import x_pca, y




def f(x):
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1 - f(x))

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y),10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect