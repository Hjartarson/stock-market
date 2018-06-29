
import os
import sys
import pandas as pd
import numpy as np
import matplotlib as plt

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
ROOT =  os.path.join(PROJ_ROOT, os.path.pardir, os.path.pardir)
DATA_RAW = os.path.join(PROJ_ROOT, "data","raw")
FIGURES = os.path.join(PROJ_ROOT, "reports","figures")

sys.path.append(os.path.join(ROOT,'modules'))

import style as style
style.set_style()
import stocks as stocks
import plot_backtest as pb

# QUOTES = ['ALFA', 'ATCO-B', 'ALIV-SDB', 'AZN', 'ASSA-B',
#           'BOL',
#           'ERIC-B', 'ELUX-B',
#           'GETI-B',
#           'HM-B',
#           'INVE-B',
#           'KINV-B',
#           'LUMI-SDB', 'LUPE',
#           'MTG-B',
#           'NDA-SEK', 'NCC-B', 'NOKIA-SEK',
#           'SKA-B', 'SAND', 'SCA-B', 'SKF-B', 'SEB-C', 'SECU-B', 'SSAB-B', 'SWED-A', 'SWMA',
#           'TELIA', 'THULE', 'TEL2-B',
#           'VOLV-B']

def get_data(quote, interval, period):
    path_quote = os.path.join(FIGURES, quote+'-'+interval+'-'+period)
    if not os.path.exists(path_quote):
        os.makedirs(path_quote)
    print('Quote:\t\t',quote)
    print('Interval:\t', interval)
    print('Period:\t\t', period)

    df = stocks.GoogleIntradayQuote(quote, interval, period)
    df = df.replace(0, np.nan).fillna(method='ffill')
    print(df.info())

    df.to_pickle(os.path.join(DATA_RAW, quote+'-'+interval+'-'+period+'.pkl'))

    f, ax = pb.create_axis()
    df[['open', 'close', 'volume']].plot(ax=ax, secondary_y='volume')

    path = os.path.join(path_quote, '01_open_close.png')
    pb.save_fig(f, path)


if __name__ == '__main__':
    quote = sys.argv[1]
    if len(sys.argv) == 2:
        interval = '1d'
        period = '1Y'
    else:
        interval = sys.argv[2]
        period = sys.argv[3]
    get_data(quote, interval, period)