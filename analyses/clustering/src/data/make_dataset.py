# -*- coding: utf-8 -*-
"""
Created on Fri May 19 07:56:31 2017

@author: admOrnHja
"""
import sys
import os

import pandas as pd
import numpy as np
import datetime as dt

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
DATA_RAW = os.path.join(PROJ_ROOT, "data", "raw")
PROCESSED = os.path.join(PROJ_ROOT, "data", "processed")
REPORTS = os.path.join(PROJ_ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")
path_components = PROJ_ROOT.split(os.sep)
ROOT = os.sep.join(path_components[:path_components.index("analyses")])

sys.path.append(os.path.join(ROOT, 'modules'))

import stocks as stocks
import style as style
style.set_style()

def make_dataset():
    #Get quotes ticks
    QUOTES = pd.read_csv(os.path.join(ROOT, 'ticks.csv'), sep='\r', squeeze=True).values
    #Get old dataset
    try:
        df_all = pd.read_pickle(os.path.join(DATA_RAW, 'df.pkl'), compression='gzip')
        mask = np.in1d(QUOTES, df_all['stock'].unique())
        QUOTES = QUOTES[~mask]
    except:
        print('no existing data')
        df_all = pd.DataFrame()

    interval = '1d'
    period = '2Y'

    today = dt.datetime.today()
    then = (today-pd.DateOffset(years=1))
    days_expected = len(pd.date_range(then, today, freq='B', tz='Etc/GMT+1'))
    print('Days Expected:',days_expected)
    quotes_data = []
    quotes_no_data = []
    quotes_missing_data = []


    for quote in QUOTES:
        print('quote:', quote)
        df_quote = stocks.GoogleIntradayQuote(quote, interval, period)
        days_obtained = df_quote.shape[0]
        print('days obtained:', days_obtained)
        if days_obtained == 0:
            print('no data')
            quotes_no_data += [quote]
        elif days_obtained/days_expected < 0.5:
            print('missing data')
            quotes_missing_data += [quote]
        else:
            print('complete data')
            df_all = df_all.append(df_quote, ignore_index=True)
            quotes_data += [quote]

    df_all.to_pickle(os.path.join(DATA_RAW, 'df.pkl'), compression='gzip')
    print('Data')
    print(quotes_data)
    print('No Data')
    print(quotes_no_data)
    print('Missing Data')
    print(quotes_missing_data)


if __name__ == '__main__':
    make_dataset()
