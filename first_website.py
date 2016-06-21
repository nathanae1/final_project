from flask import Flask, request
import pandas as pd
from collections import Counter
import ast
app = Flask(__name__)


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


def str_list_to_series(text):
    return pd.Series(ast.literal_eval(text), dtype=float)


# Form page to submit text
@app.route('/')
def submission_page():
    return '''
        <form action="/results" method='POST' >
            Model Predictions:
            <input type="text" name="predictions" />
            <br>Actual Values:
            <input type="text" name="actual" />
            <br>Category to split on:
            <input type="text" name="categorical" />
            <input type="submit" />
        </form>
        '''


@app.route('/results', methods=['POST'])
def api_results():
    predictions = [float(x) for x in
                   ast.literal_eval(str(request.form['predictions']))]
    actual = str_list_to_series(str(request.form['actual']))
    categorical = str_list_to_series(str(request.form['categorical']))
    out_dict = test_variable_numer(predictions, actual, categorical)
    out_str = '''
              True estimates were off by: {0} <br>
              False estimates were off by: {1} <br>
              '''.format(out_dict['True'], out_dict['False'])
    return out_str


# My word counter app
# @app.route('/word_counter', methods=['POST'])
# def word_counter():
#     text = str(request.form['user_input'])
#     word_counts = Counter(text.lower().split())
#     page = 'There are {0} words.<br><br>Individual word counts:<br> {1}'
#     return page.format(len(word_counts), dict_to_html(word_counts))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
