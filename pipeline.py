import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from scipy.stats import ks_2samp, chisquare, ttest_ind


class feature_remover():
    def __init__(self, X, y, removed_feature, model=RandomForestRegressor,
                 split=.3):
        '''
        INPUT: X: pd.DataFrame - Matrix to be predicted on (does not contain
                                 removed feature)
               y: pd.series - predicted variable
               removed_feature: pd.series - feature to be removed from data
               model: model - used to make predictions,
                              default RandomForestRegressor
               split: float between 0.0 and 1.0 - size of train/test split
        '''
        self.X = X
        self.y = y
        self.removed_feature = removed_feature
        self.model = model
        self._split = split
        self._train_test_split = train_test_split(split=split)
        self.cleaned_X = self._feature_removal_pipeline(X, removed_feature)

    def _feature_removal_pipeline(df, removed_feature):
        '''
        INPUT:  removed_feature: pd.Series (not already in df)
                df: pd.DataFrame
        OUTPUT: df: pd.DataFrame
        Removes the feature present in column from the df by performing PCA on
        the data taking the inputted column as your principle component.
        '''
        rm_feat_arr = np.array(removed_feature)
        return_mat = []
        for column in df.columns:
            col_arr_df = np.array(df[column])
            return_mat.append(col_arr_df -
                              calculate_linalg_beta(col_arr_df, rm_feat_arr) *
                              rm_feat_arr)
        return pd.DataFrame(np.array(return_mat).T, columns=df.columns)

    def _calculate_linalg_beta(X1, X2):
        '''
        INPUT:  X1: np.array, array to be transformed
                X2: np.array, transforming array
        OUTPUT: X1_prime: np.array
        Calculates the beta coefficents for and then removes the influence of
        X1 from X2.
        '''
        X1_mean = X1.mean()
        X2_mean = X2.mean()
        beta = (X2-X2_mean).T*(X1-X1_mean)/((X2-X2_mean).T*(X2-X2_mean))
        X1_prime = X1 - beta*X2
        return X1_prime
