
import os
import sys
import pandas as pd
import numpy as np


CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
ROOT =  os.path.join(PROJ_ROOT, os.path.pardir, os.path.pardir)
DATA_IM = os.path.join(PROJ_ROOT, "data","intermediate")
DATA_PRED = os.path.join(PROJ_ROOT, "data","prediction")
sys.path.append(os.path.join(ROOT,'modules'))
FIGURES = os.path.join(PROJ_ROOT, "reports","figures")

import style as style
style.set_style()
import plot_backtest as pb
from backtesting import BackTesting as BT

def make_pred(quote, interval, period):
    file = 'pred-' + quote + '-' + interval + '-' + period + '.pkl'
    print('loading file:', file)
    df = pd.read_pickle(os.path.join(DATA_PRED, file))
    print(df.tail(20))
    #df['prediction_bool'] = df['prediction'].apply(pred_to_bin)

    outcome = [x for x in df.columns if x.startswith('y_')][0]
    predictions = [x for x in df.columns if x != outcome]
    print(outcome)
    print(predictions)
    for pred in predictions:
        df[pred+'_bool'] = df[predictions].apply(pred_to_bin,axis=1)
        df[pred + '_return'] = df[pred+'_bool'].multiply(df[outcome], axis='index')

    predictions = [s + '_return' for s in predictions]
    df['long'] = df[outcome]
    df['short'] = df[outcome]*-1
    print(df.tail(20))
    f, ax = pb.create_axis()
    df[['long', 'short']+predictions].dropna().cumsum().plot(ax=ax)
    path = os.path.join(FIGURES, 'return.png')
    pb.save_fig(f, path)
    print(df[['long', 'short'] + predictions].dropna().sum())


def pred_to_bin(Series):
    Series[Series >= 0.8] = 1
    Series[Series < 0.2] = -1
    Series[(Series < 1) & (Series > 0)] = 0
    return Series

# def pred_to_bin(df, pred):
#     if pred >= 0.7:  # Long
#         prediction = 1
#     elif pred < 0.3:  # Short
#         prediction = -1
#     else:  # Stay
#         prediction = 0
#     return prediction

if __name__ == '__main__':
    quote = sys.argv[1]
    if len(sys.argv) == 2:
        interval = '1d'
        period = '1Y'
    else:
        interval = sys.argv[2]
        period = sys.argv[3]
    make_pred(quote, interval, period)