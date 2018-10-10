import os
import shutil
import yaml

import pandas as pd

def load_dataset(year, shuffle=False):
    """Loads chosen data set, mixes it and returns."""
    main_path = "C:/Users/Derrick/Desktop/Masters Thesis/Data/"
    file_name = '{}year.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path, na_values='?')
    Y = df['class'].values
    X = df.drop('class', axis=1).values
    if shuffle:
        shuffled_idx = np.random.permutation(len(Y))
        X = X[shuffled_idx, :]
        Y = Y[shuffled_idx]
    return X, Y


def load_data():
    # 1year_no_heading_zeros.csv OR 1_Year_Training_Set_Equalized.csv
    df = pd.read_csv(
        filepath_or_buffer=r"C:\Users\Derrick\Desktop\Masters Thesis\data\1year.csv",
        header=None,
        sep=',')

    ''' DECLARE INDEPENDENT VARIABLE NAMES IN THE FOLLOWING FORMAT 
    df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
                  'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20',
                  'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30',
                  'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40',
                  'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50',
                  'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60',
                  'X61', 'X62', 'X63', 'X64', 'Class']
                  
    # Reduced dataset based on Altman's five ratios
    df.columns = ['working capital / total assets',
                  'retained earnings / total assets',
                  'EBIT / total assets',
                  'book value of equity / total liabilities',
                  'sales / total assets',
                  'class']
    '''

    # Original dataset columns (all columns) UNHIDE FOR COMPLETE DATASET
    df.columns = ['net profit / total assets', 'total liabilities / total assets', 'working capital / total assets', 'current assets / short-term liabilities',
                  '[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365', 'retained earnings / total assets',
                  'EBIT / total assets', 'book value of equity / total liabilities', 'sales / total assets', 'equity / total assets', '(gross profit + extraordinary items + financial expenses) / total assets',
                  'gross profit / short-term liabilities', '(gross profit + depreciation) / sales', '(gross profit + interest) / total assets', '(total liabilities * 365) / (gross profit + depreciation)',
                  '(gross profit + depreciation) / total liabilities', 'total assets / total liabilities', 'gross profit / total assets', 'gross profit / sales', '(inventory * 365) / sales',
                  'sales (n) / sales (n-1)', 'profit on operating activities / total assets', 'net profit / sales', 'gross profit (in 3 years) / total assets', '(equity - share capital) / total assets',
                  '(net profit + depreciation) / total liabilities', 'profit on operating activities / financial expenses', 'working capital / fixed assets', 'logarithm of total assets',
                  '(total liabilities - cash) / sales', '(gross profit + interest) / sales', '(current liabilities * 365) / cost of products sold', 'operating expenses / short-term liabilities',
                  'operating expenses / total liabilities', 'profit on sales / total assets', 'total sales / total assets', '(current assets - inventories) / long-term liabilities', 'constant capital / total assets',
                  'profit on sales / sales', '(current assets - inventory - receivables) / short-term liabilities', 'total liabilities / ((profit on operating activities + depreciation) * (12/365))',
                  'profit on operating activities / sales', 'rotation receivables + inventory turnover in days', '(receivables * 365) / sales', 'net profit / inventory',
                  '(current assets - inventory) / short-term liabilities', '(inventory * 365) / cost of products sold', 'EBITDA (profit on operating activities - depreciation) / total assets',
                  'EBITDA (profit on operating activities - depreciation) / sales', 'current assets / total liabilities', 'short-term liabilities / total assets', '(short-term liabilities * 365) / cost of products sold)',
                  'equity / fixed assets', 'constant capital / fixed assets', 'working capital', '(sales - cost of products sold) / sales',
                  '(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)', 'total costs /total sales', 'long-term liabilities / equity', 'sales / inventory',
                  'sales / receivables', '(short-term liabilities *365) / sales', 'sales / short-term liabilities', 'sales / fixed assets', 'Class']

    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    # split data table into data X and class labels Y
    x = df.ix[:, 0:64].values
    y = df.ix[:, 64].values

    return x, y




