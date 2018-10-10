import pandas as pd
import numpy as np
import time
from datetime import timedelta
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from PredictBankruptcy import X_reduced, Y

start_time = time.time() # Start recording time of program

X_scale = StandardScaler()
X = X_scale.fit_transform(X_reduced)

from sklearn.model_selection import train_test_split
y = Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y),2))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

y_train[0], y_v_train[0]

nn_structure = [7, 7, 7, 2]

def f(x):
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1 - f(x))

import numpy.random as rand
def setup_and_init_weights(nn_structure):
    W = {}
    b ={}
    for l in range(1, len(nn_structure)):
        W[l] = rand.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = rand.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def feed_forward(x, W, b):
    h = {1: x}
    z ={}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(n1) = -(y_i - h_i^(n1) * f'(z_i^(n1))
    return -(y-h_out)*f_deriv(z_out)

def calculate_hidden_delta(delta_plus_l, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_l) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=1000, alpha=0.1):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values
            # to be used in the gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from n1-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:],h[l],z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis],np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure)-1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration Number')
plt.show()

elapsed_time_secs = time.time() - start_time # Calculate elapsed time
msg = "Execution took: %s secs (wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

print(msg)

def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 4)
accuracy_score(y_test, y_pred)*100