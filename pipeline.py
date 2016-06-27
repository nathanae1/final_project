import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from scipy.stats import ks_2samp, chisquare, ttest_ind


class feature_remover(object):
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
        self.transformed_X = self._feature_removal_pipeline(X, removed_feature)
        self._untransformed_X_train, self._untransformed_X_test,
        self._transformed_X_train, self._transformed_X_test
        self._y_train, self._y_test,
        self._removed_feature_train, self._removed_feature_test =
        self._train_test_split(self.X, self.transformed_X,
                               self.y, self.removed_feature)
        self.untransformed_model = None
        self.transformed_model = None
        self.untransformed_predictions = None
        self.transformed_predictions = None

    def fit(self):
        '''
        INPUT:  None
        OUTPUT: None
        Fits both the transformed and untransformed data
        '''
        self.untransformed_model = self.model()
        self.transformed_model = self.model()

        print 'Fitting model on the raw data...'
        self.untransformed_model.fit(self._untransformed_X_test, self._y_train)
        print 'Fitting model on the transformed data'
        self.transformed_model.fit(self._transformed_X_train, self._y_train)
        print 'Finished!'

    def predict(self):
        '''
        INPUT:  None
        OUTPUT: None
        Predicts the test responses on both the untransformed and transformed
        data
        '''
        self.untransformed_predictions = \
            self.untransformed_model.predict(self._untransformed_X_test)

        self.transformed_predictions = \
            self.transformed_model.predict(self._transformed_X_test)

    def scores(self):
        scores = dict({})
        scores['Untransformed'] = \
            _test_variable_numer(self.untransformed_predictions,
                                 self._y_test,
                                 self._removed_feature_test
                                 )
        scores['Transformed'] = \
            _test_variable_numer(self.transformed_predictions,
                                 self._y_test,
                                 self._removed_feature_test
                                 )
        return scores

    def _feature_removal_pipeline(self, df, removed_feature):
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
                              self._calculate_linalg_beta(col_arr_df,
                                                          rm_feat_arr) *
                              rm_feat_arr)
        return pd.DataFrame(np.array(return_mat).T, columns=df.columns)

    def _calculate_linalg_beta(self, X1, X2):
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

    def _test_variable_numer(self, predictions, actual, categorical):
        '''
        INPUT: list predictions: list of a model's predictions
               pandas series actual: series of the actual values
               pandas series categorical: pandas series of to which category
               each row prediction/value pair belongs
        OUTPUT: dict {'False':float, 'True':float}
               Where True and False equate to the predictions and True %IQR and
               False %IQR are the raw differences for True and False divided by
               the IQR of the difference between the actual values and the
               predictions
        '''
        difference = pd.DataFrame(predictions-actual)
        diff_iqr = difference.quantile(.75) - difference.quantile(.25)

        # Calculating differences for 0:
        diff_0 = difference[categorical == 0].mean()

        # Calculating differences for 1:
        diff_1 = difference[categorical == 1].mean()

        length = min([difference[categorical == 0].shape[0],
                      difference[categorical == 1].shape[0]])

        k_score = ks_2samp(np.array(difference[categorical == 0]
                           [:length]).reshape(-1,),
                           np.array(difference[categorical == 1]
                           [:length]).reshape(-1,))

        chi_score = chisquare([np.array(difference[categorical == 0]
                              [:length]).reshape(-1,),
                              np.array(difference[categorical == 1]
                                       [:length]).reshape(-1,)],
                              axis=1)

        t_score = ttest_ind(np.array(difference[categorical == 0])
                            .reshape(-1,),
                            np.array(difference[categorical == 1])
                            .reshape(-1,))

        return dict({'False': diff_0[0],
                     'True': diff_1[0],
                     'k-score': k_score,
                     'chi2-score': chi_score,
                     't-score': t_score
                     }
                    )
