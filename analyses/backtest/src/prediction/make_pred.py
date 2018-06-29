
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

import plot_backtest as pb

def make_pred(quote, interval, period):
    file = 'feat-' + quote + '-' + interval + '-' + period + '.pkl'
    print('loading file:', file)
    df = pd.read_pickle(os.path.join(DATA_IM, file))
    print(df.head())
    from backtesting import BackTesting as BT
    from machine_learning import MachineLearning as ML
    ml = ML()

    xvar = [x for x in df.columns if x.startswith('x_')]
    pred_var = 'y_open_close_days_' + str(4)
    pred_var_bool = pred_var + '_up'

    result = df[pred_var]
    clfs = ['xg_boost',
            #'random_forrest',
            #'perceptron',
            #'gradient_boost'
            ]
    for clf in clfs:

        mla = ml.set_classifier(clf)
        bt = BT(clf=mla, data=df[xvar+[pred_var_bool]])
        bt.set_output_variable(pred_var_bool)
        bt.run_backtest(start_row=160)
        df_pred = bt.get_result().dropna()
        pred = df_pred.rename(clf)

        result = pd.concat([result, pred], axis=1)

    result.to_pickle(os.path.join(DATA_PRED, 'pred-'+quote + '-' + interval + '-' + period + '.pkl'))
    print(result.tail(40))

    # f, ax = pb.create_axis()
    # pred['accuracy'].cumsum().plot(ax=ax)
    # path = os.path.join(FIGURES, 'accuracy.png')
    # pb.save_fig(f, path)

if __name__ == '__main__':
    quote = sys.argv[1]
    if len(sys.argv) == 2:
        interval = '1d'
        period = '1Y'
    else:
        interval = sys.argv[2]
        period = sys.argv[3]
    make_pred(quote, interval, period)