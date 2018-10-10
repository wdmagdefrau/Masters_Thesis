# Refer https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# Refer https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Aim of this is to show case use of keras and grid search libraries

from __future__ import print_function
import tensorflow as tf
import sys
import os
import math
import time
import numpy as np
import matplotlib.pylab as plt
from datetime import timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from data_processors import Imputer, Processor
from utils import split_dataset, load_yaml_and_save, load_dataset

#print(__doc__)

# Create list of parameters to be used in param_grid for grid search in GridSearchCV - lists must be added to param_grid for cross-validation
activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax', 'softplus', 'softsign']
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.2, 0.5, 0.8]
weight_constraint = [1, 2, 3, 4, 5]
neurons = [250, 500]
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

def main(yaml_path='./config.yml', run_name=None):
    # Create output directory where experiment is saved
    if run_name is None:
        run_name = time.strftime('%Y%m%d-%H%M', time.localtime())
    run_path = os.path.join('./output', run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)  # TODO need to close this loop and make sure results are being saved for future analysis and output recovery
    config = load_yaml_and_save(yaml_path, run_path)
    # Fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # Load dataset using utils.load_dataset({year})
    X, Y = load_dataset(1)
    #n_samples = len(X)
    # TODO improve training by altering how the data is sampled - not sure where this will need to be placed
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    processor = Processor(**config['processor_params'])
    X_train = processor.fit_transform(X_train)
    X_test = processor.transform(X_test)
    imputer = Imputer(**config['imputer_params'])
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    input_dim = X_train.shape[1] # number of columns

    scores = ['precision', 'recall']

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)  # use verbose=1 if you want to see progress
    epochs = [10, 25]  # add 50, 100, 150 etc
    batch_size = [1000, 5000]  # add 5, 10, 20, 40, 60, 80, 100 etc
    param_grid = dict(epochs=epochs, batch_size=batch_size)

    for score in scores:
        print("########################################################################################")
        print("# Tuning hyper-parameters for %s" % score)
        print()
        # Start recording run-time of program
        start_time = time.time()
        # create grid of parameter models and perform experiment
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=10,
                            scoring='%s_macro' % score,
                            n_jobs=-1)
        grid_result = dict()
        # TODO add validation data to .fit(X_train, Y_train, [WHERE?/HOW?]) so that we can use binary crossentropy as a metric in create_model() .compile
        grid_result['fit_info'] = grid.fit(X_train, Y_train, validation_split=0.3)
        print('\nGrid Results: \n\t%s' % grid_result)

        grid_pred_proba = grid.predict_proba(X_test)
        grid_pred = np.argmax(grid_pred_proba, axis=1)

        print('\nBest parameters found from development / training set: ')
        print()
        print('\t%s' % grid.best_params_)
        print()
        print('Grid scores on development / training set: ')
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print('Detailed Classification Report:')
        print()
        print('\tThe model is trained on the full development set.')
        print('\tScores are computed on the full evaluation set.')
        print()
        #y_true, y_pred = Y_test, grid.predict(X_test)  # Original code
        y_true, y_prob = Y_test, grid.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        # TODO Unhide dictionary declarations to store that particular metric - currently turned off until run-time can be tested
        # Calculate and save results
        #grid_result['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
        #grid_result['brier_score_loss'] = metrics.brier_score_loss(y_true, y_prob[:,1])
        #grid_result['f1_score'] = metrics.f1_score(y_true, y_pred, labels=[0, 1])
        #grid_result['hamming_loss'] = metrics.hamming_loss(y_true,y_pred)
        #grid_result['jaccard_similarity_score'] = metrics.jaccard_similarity_score(y_true, y_pred)
        #grid_result['zero_one_loss'] = metrics.zero_one_loss(y_true, y_pred)
        #grid_result['precision_recall_curve'] = metrics.precision_recall_curve(y_true, y_prob[:,1])
        #grid_result['log_loss'] = metrics.log_loss(y_true, y_prob[:, 1])
        #grid_result["matthews_correlation_coefficient"] = metrics.matthews_corrcoef(y_true, y_pred)
        grid_result['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        #grid_result['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        grid_result['recall'] = metrics.recall_score(y_true, y_pred, labels=[0, 1])
        #grid_result['precision'] = metrics.precision_score(y_true, y_pred, labels=[0, 1])
        #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
        #grid_result['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        #grid_result['roc_auc'] = metrics.auc(fpr, tpr)
        grid_result['classification_report'] = metrics.classification_report(y_true, y_pred, labels=[0, 1])
        print(classification_report(y_true, y_pred))
        print()

        #grid_pred_proba = grid.predict_proba(X_test)
        #grid_pred = np.argmax(grid_pred_proba, axis=1)

        ##############################################################
        # calculate and print run-time
        elapsed_time_secs = time.time() - start_time  # Calculate elapsed time
        msg = "Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs))
        print(msg)


########################################################
# Function to create model, required for KerasClassifier
def create_model(input_dim=110):  # TODO need a way for input_dim to change with the data set pre-processing AND determine if 1-hot needs reduction
    #print('\nModel Input Dimensions: %s' % input_dim)
    # default values for initialization
    activation = 'relu'
    dropout_rate = 0.0
    init_mode = 'uniform'
    weight_constraint = 0
    optimizer = 'adam'
    lr = 0.01
    momentum = 0
    #print("TEST: Input Dimension: %s" % input_dim)
    # create model layers given parameter inputs
    model = Sequential()
    model.add(Dense(input_dim,
                    input_dim=input_dim, kernel_initializer=init_mode,
                    activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(250, activation=activation, kernel_initializer='uniform'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'binary_crossentropy'])
    #print('\nModel Summary: \n%s' % model.summary())  # remove octathorpe to print model summary
    return model


if __name__ == '__main__':
    try:
        yaml_path = r'C:\Users\Derrick\Desktop\Masters Thesis PyCharm Project V2\Bankruptcy_Prediction\configs\config_mlp.yml'
    except IndexError as e:
        print('You have to specify the config.yaml to use as `python run.py '
            'example_config.yaml`')
        print('Exiting.')
        sys.exit()
    main(yaml_path=yaml_path)
