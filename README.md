# Better understanding the idea of making equal models:
## Day 1:
I wanted to get a test data set clean today and do a proof of concept on a
simple model. I spent the morning cleaning the data, which took longer then
expected since all of the information for each row was contained in a single
string of text with no spaces. Afterwards, I tested (and tuned) 3 models, of
which the Random Forest did the best (a whopping .24 R^2) made a simple pipeline
(explained later) which I passed in the predictions, actual values, and an array
of whether or not the respondent was white (which I had withheld from the
equation during model testing). Surprisingly the model actually predicted quite
a bit differently whether or not a person was white, with it underestimating the
wages of non-whites by $8,875 and overestimating white wages by $5,142. For a
model that was just supposed to give a simple proof of concept, the extreme
differences actually somewhat worry me, if differences are this easy to extract,
how many production models are there out there that might fall victim to these
problems?


_In an effort to minimize the affect of minorities possibly making less than white people in general, I split the dataset in two and compared the average difference between predicted income and actual income for each. I passed my predictions, the actual incomes and a boolean array of whether or not a person self-reported as being white in the census data into the array._
```p
def test_variable_numer(predictions, actual, categorical):
    '''
    INPUT: list predictions: list of a model's predictions
           pandas series actual: series of the actual values
           pandas series categorical: pandas series of to which category each
              row prediction/value pair belongs
    OUTPUT: dict {0:predicted_0, 1:predicted_1}
    '''
    # Finding the difference between the actual and predicted values
    difference = pd.DataFrame(actual - predictions)

    # Calculating differences for False:
    diff_0 = difference[categorical == 0].mean()

    # Calculating differences for True:
    diff_1 = difference[categorical == 1].mean()

    return dict({'False': diff_0[0], 'True': diff_1[0]})
```

## Day 2:
Today was the first day of using preprocessing to remove unwanted signal from
the data. After talking to my instructors, the two preprocessors that emerged as
the best were using a linear transform on the variables to move them to an
orthogonal plane to the unwanted variable and doing PCA with the variable as the
principle component to achieve the same results but with more rigorous
constraints. Both the models made the census data predictions more equal across
the two groups, but the PCA model was able to maintain almost the same level of
prediction quality as the original model.

_The better model_
```p
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
```
_The worse model (*DO NOT USE*)_
```
def feature_removal_pipeline(column, df):
    '''
    INPUT:  df: pd.DataFrame
            column: pd.Series (not already in df)
    OUTPUT: df: pd.DataFrame
    '''
    col_arr = np.array(column)
    mag = np.sqrt(col_arr.dot(col_arr))
    unit_col_arr = col_arr/mag
    unit_col_mat = (1-unit_col_arr*unit_col_arr.T)
    return_mat = []
    for column in df.columns:
        col_arr_df = np.array(df[column])
        return_mat.append(unit_col_mat*col_arr_df)
    return pd.DataFrame(return_mat, columns=df.columns)
```

## Day 3:
Today was spent on planning and nomenclature. I expected preprocessing the data
to take longer than it did, so I spent quite a while thinking about how to
better my models or reporting. I eventually came to the decision to update my
similarity predictor to use the Kolmogorov-Smirnov statistic as a possible
metric for distributional differences in the data. In order to double check that
the scipy implementation was actually capable of discerning differences in shape
in 'real world' data, I used a brute force method of realigning the means (just
subtracting the differences out) as a baseline to compare against.

## Day 4:
I spent most of today trying to figure out how to best report the distributional
differences that splitting the data creates. T-tests are the be the most
consistent way to compare the distributions as of right now, but I am leaving
other metrics in to make the scoring function more generalizable.
