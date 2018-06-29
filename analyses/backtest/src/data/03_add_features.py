
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
ROOT =  os.path.join(PROJ_ROOT, os.path.pardir, os.path.pardir)
DATA_RAW = os.path.join(PROJ_ROOT, "data","raw")
DATA_IM = os.path.join(PROJ_ROOT, "data","intermediate")
sys.path.append(os.path.join(ROOT,'modules'))
FIGURES = os.path.join(PROJ_ROOT, "reports","figures")

import style as style
style.set_style()
import features as feat
import plot_backtest as pb

def add_features(quote, interval, period):
    path_quote = os.path.join(FIGURES, quote+'-'+interval+'-'+period)
    if not os.path.exists(path_quote):
        os.makedirs(path_quote)
    file = quote + '-' + interval + '-' + period + '.pkl'
    print('loading file:', file)
    df = pd.read_pickle(os.path.join(DATA_RAW, file))
    # ADD IN
    #Shift volume column due to not known volume for full day at opening
    df['x_volume'] = df['volume'].shift(1).copy()

    df = df.pipe(feat.add_rolling)
    df = df.pipe(feat.add_x_days_change)


    x_var = [x for x in np.unique(df.columns) if x.startswith('x_')]
    # NORMALIZE
    # df[x_var] = df[x_var].pipe(normalize)
    print(df.tail())
    # df = df.pipe(feat.shift_columns, x_var, 10)

    # ADD OUT
    df = df.pipe(feat.add_outcome, days=10)
    y_var = [x for x in np.unique(df.columns) if x.startswith('y_')]
    df.to_pickle(os.path.join(DATA_IM, 'feat-'+quote + '-' + interval + '-' + period + '.pkl'))

    # f, ax = pb.create_axis()
    # plt.imshow(df[x_var].values, cmap='hot', interpolation='nearest', aspect='equal')
    # path = os.path.join(path_quote, '03_features.png')
    # pb.save_fig(f, path)

def normalize(df):
    return df.sub(df.dropna().mean(), axis=1).div(df.dropna().std(), axis=1)

if __name__ == '__main__':
    quote = sys.argv[1]
    if len(sys.argv) == 2:
        interval = '1d'
        period = '1Y'
    else:
        interval = sys.argv[2]
        period = sys.argv[3]
    add_features(quote, interval, period)