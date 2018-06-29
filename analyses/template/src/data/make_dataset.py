# -*- coding: utf-8 -*-
"""
Created on Fri May 19 07:56:31 2017

@author: admOrnHja
"""
import sys
import os

import pandas as pd
import time

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

def make_dataset(quote):
    print('Getting data for', quote)
    df = stocks.GoogleIntradayQuote(quote, '1d', '1Y')
    print(df.head())
    df.to_pickle(os.path.join(DATA_RAW,'df.pkl'), compression='gzip')
        
if __name__ == '__main__':
    make_dataset(sys.argv[1])
