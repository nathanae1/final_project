from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare, ttest_ind
from collections import Counter
import ast
app = Flask(__name__)


def _test_variable_numer(predictions, actual, categorical):
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
    difference = pd.DataFrame(predictions - actual)
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


def str_list_to_series(text):
    return pd.Series(ast.literal_eval(text), dtype=float)


# Form page to submit text
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/_results')
def results():
    predictions = request.args.get('predictions', '', type=str)
    actual = request.args.get('actual', '', type=str)
    categorical = request.args.get('categorical', '', type=str)
    predictions = [float(x) for x in
                   ast.literal_eval(predictions)]
    actual = str_list_to_series(actual)
    categorical = str_list_to_series(categorical)
    out_dict = _test_variable_numer(predictions, actual, categorical)
    out_str = '''
              True estimates were off by: {0} <br>
              False estimates were off by: {1} <br>
              T-score: statistic: {2} p value: {3}<br>
              K-score: statistic: {4} p value: {5}<br
              '''.format(out_dict['True'],
                         out_dict['False'],
                         out_dict['t-score'][0],
                         out_dict['t-score'][1],
                         out_dict['k-score'][0],
                         out_dict['k-score'][1],)
    return jsonify(false_mean=out_dict['False'],
                   true_mean=out_dict['True'],
                   t_test_p_value=out_dict['t-score'][1])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
