import datetime
from datetime import timedelta
import pandas as pd
import plotly.graph_objs as go

fname = 'sarima_test_predictions.csv'


def filter_data(data, start_date, end_date):
    if type(end_date) is not datetime.datetime:
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    if type(end_date) is not datetime.datetime:
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    if start_date == end_date:
        end_date = end_date + timedelta(days=1)
    data = data[(data.index >= start_date) & (data.index < end_date)]
    return data


def get_production(day, expected=True):
    predictions = pd.read_csv(fname, sep=',')
    predictions['TIMESTAMP'] = [pd.to_datetime(x) for x in predictions['TIMESTAMP']]
    predictions.set_index('TIMESTAMP', drop=True, inplace=True)
    predictions = filter_data(predictions, day, day)
    predictions = [round(x * 1000, 3) for x in list(predictions['y_pred'])] \
        if expected else [round(x * 1000, 3) for x in list(predictions['y_true'])]
    return [0 if x < 0 else x for x in predictions]
    # return values in Wh
