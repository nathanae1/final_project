# Pipeline layout:
# Peel off column to be predicted on
# Perform calculations on each individual column

import pandas as pd
import numpy as np


# THIS IS WRONG DO NOT USE

# def feature_removal_pipeline(column, df):
#     '''
#     INPUT:  df: pd.DataFrame
#             column: pd.Series (not already in df)
#     OUTPUT: df: pd.DataFrame
#     '''
#     col_arr = np.array(column)
#     mag = np.sqrt(col_arr.dot(col_arr))
#     unit_col_arr = col_arr/mag
#     unit_col_mat = (1-unit_col_arr*unit_col_arr.T)
#     return_mat = []
#     for column in df.columns:
#         col_arr_df = np.array(df[column])
#         return_mat.append(unit_col_mat*col_arr_df)
#     return pd.DataFrame(return_mat, columns=df.columns)


def feature_removal_pipeline(df, column):
    '''
    INPUT:  df: pd.DataFrame
            column: pd.Series (not already in df)
    OUTPUT: df: pd.DataFrame
    Removes the influence of column from the DataFrame
    '''
    # Convert to a numpy array for easier data manipulation
    col_arr = np.array(column)
    return_mat = []

    # For every column in the matrix, perform the transformation by calling
    # calculate_linalg_beta
    for column in df.columns:
        return_mat.append(calculate_linalg_beta(df[column], col_arr))

    # Returns the processed data in a dataframe with the same formatting as the
    # df that was initially entered.
    return pd.DataFrame(return_mat, columns=df.columns)


def calculate_linalg_beta(X1, X2):
    '''
    INPUT:  X1: np.array, array to be transformed
          X2: np.array, transforming array
    OUTPUT: X1_prime: np.array
    Calculates the correlation between X1 and X2 and returns X1 with the
    correlation removed.
    '''

    X1_mean = X1.mean()
    X2_mean = X2.mean()
    beta = (X2-X2_mean).T*(X1-X1_mean)/((X2-X2_mean).T*(X2-X2_mean))
    X1_prime = X1 - beta*X2
