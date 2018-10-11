# Refer https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# Refer https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Aim of this is to show case use of keras and grid search libraries

from __future__ import print_function
import tensorflow as tf
import sys
import math
import time
from datetime import timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from data_processors import Imputer, Processor
from utils import split_dataset, load_yaml_and_save, load_dataset

#from PredictBankruptcy import x_comp, y
#from sklearn.preprocessing import StandardScaler
#from keras.optimizers import SGD

scores = ['accuracy','precision', 'recall']

def main(yaml_path='./config.yml', run_name=None):
    # Start recording run-time of program
    start_time = time.time()
    # Create output directory where experiment is saved
    if run_name is None:
        run_name = time.strftime('%Y%m%d-%H%M', time.localtime())
    run_path = os.path.join('./output', run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    config = load_yaml_and_save(yaml_path, run_path)
    # Fix random seed for reproducibility
    seed = 7456
    np.random.seed(seed)
    # Load dataset using utils.load_dataset({year})
    X, Y = load_dataset(1)
    # TODO improve training by altering how the data is sampled - not sure where this will need to be placed
    X_train, Y_train, X_test, Y_test = split_dataset(X, Y,
                                                     config['experiment']['test_share']) # TODO is this the best "split" function to use? Look into alternative used in single trial cross validation (train_test_split{})

    processor = Processor(**config['processor_params'])
    X_train = processor.fit_transform(X_train)
    X_test = processor.transform(X_test)
    imputer = Imputer(**config['imputer_params'])
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    input_dim = X_train.shape[1] # number of columns

    # TODO add list of scorers and iterate over list with for loop, calling the experiment each time

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)  # use verbose=1 if you want to see progress
    epochs = [10, 25]  # add 50, 100, 150 etc
    batch_size = [1000, 5000]  # add 5, 10, 20, 40, 60, 80, 100 etc
    param_grid = dict(epochs=epochs, batch_size=batch_size)
    # create a dictionary to store results
    grid_result = dict()
    # create grid of parameter models and perform experiment
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)  # scoring='f1_weighted'
    #  TODO add validation data to .fit(X_train, Y_train, [WHERE?/HOW?]) so that we can use binary crossentropy as a metric in create_model() .compile
    #  TODO potential fix: alter the shapes of the datasets so that they are all identical: X_
    #  TODO potential fix: remove the extra list from inside of np.array() when defining X_train
    # fit_time_start = time.time()
    grid_result['fit_info'] = grid.fit(X_train, Y_train, validation_data=(X_test, Y_test))
    y_true, y_prob = Y_test, grid.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # print prediction results for comparison
    #print('\nY Test: \n%s' % Y_test)
    #print('\nPrediction: \n%s' % grid_pred)

    # Calculate and save results
    grid_result['log_loss'] = metrics.log_loss(y_true, y_prob[:, 1])
    grid_result['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    grid_result['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    grid_result['recall'] = metrics.recall_score(y_true, y_pred, labels=[0, 1])
    grid_result['precision'] = metrics.precision_score(y_true, y_pred, labels=[0, 1])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
    grid_result['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    grid_result['roc_auc'] = metrics.auc(fpr, tpr)
    grid_result['classification_report'] = metrics.classification_report(y_true,
                                                                         y_pred, labels=[0, 1])

    #confusion_matrix = results['confusion_matrix']

    tn = grid_result['confusion_matrix'][0][0]
    tp = grid_result['confusion_matrix'][1][1]
    fp = grid_result['confusion_matrix'][0][1]
    fn = grid_result['confusion_matrix'][1][0]

    #mcc = ((tp * tn) - (fp * fn))/math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

    #print('\n\nLog Loss / Accuracy: %s' % results['log_loss'], results['accuracy'])
    print("\n#############################################################################")
    print('\nConfusion Matrix: \n%s' % grid_result['confusion_matrix'])
    print('\nTrue Negatives: %s' % tn)
    print('True Positives: %s' % tp)
    print('False Positives: %s' % fp)
    print('False Negatives: %s' % fn)
    #print("\nMatthew's Correlation Coefficient: %s" % mcc )
    #print("False-Positive Rate: %s" % fpr)
    #print("True-Positive Rate: %s" % tpr)
    print("\n#############################################################################")
    #print('\nClassification Report: \n%s' % results['classification_report'])

    ##############################################################
    # TODO summarize results in a separate callable function
    print("\nBest Score: %f using %s\n\t" % (grid_result['fit_info'].best_score_, grid_result['fit_info'].best_params_))
    mean_test_score = grid_result['fit_info'].cv_results_['mean_test_score']
    test_stds = grid_result['fit_info'].cv_results_['std_test_score']
    params = grid_result['fit_info'].cv_results_['params']
    for mean_test, test_stdev, param in zip(mean_test_score,
                                            test_stds,
                                            params):
        print("%f (%f) with: %r" % (mean_test, test_stdev, param))

    ##############################################################
    # calculate and print run-time
    elapsed_time_secs = time.time() - start_time # Calculate elapsed time
    msg = "Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs))
    print("\n%s" % msg)


########################################################
# Create param_grid lists for grid search in GridSearchCV function, below
activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax', 'softplus', 'softsign']
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.2, 0.5, 0.8]
weight_constraint = [1, 2, 3, 4, 5]
neurons = [250, 500]
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']


########################################################
# Function to create model, required for KerasClassifier
def create_model(input_dim=113):  # TODO need a way for input_dim to change with the data set pre-processing AND determine if 1-hot needs reduction
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
    #print('\nModel Summary: \n%s' % model.summary())  # unhide to print summary output of NN structure
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



