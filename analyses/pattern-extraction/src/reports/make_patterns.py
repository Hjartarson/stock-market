# -*- coding: utf-8 -*-
"""
Created on Fri May 19 07:56:31 2017

@author: admOrnHja
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
DATA_RAW = os.path.join(PROJ_ROOT, "data", "raw")
PROCESSED = os.path.join(PROJ_ROOT, "data", "processed")
REPORTS = os.path.join(PROJ_ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")
path_components = PROJ_ROOT.split(os.sep)
ROOT = os.sep.join(path_components[:path_components.index("analyses")])


sys.path.append(os.path.join(ROOT, 'modules'))

import autoencoder as ae
import style as st
st.set_style()

def make_report():
    print('making report')
    df = pd.read_pickle(os.path.join(DATA_RAW, 'df.pkl'), compression = 'gzip')

    df_stock = df.groupby(['datetime', 'stock'])['close'].sum().unstack().ffill().bfill()
    df_stock = df_stock.rolling(1).mean().dropna()
    #df_stock = np.exp(df_stock)
    df_stock = (df_stock - df_stock.min()) / (df_stock.max() - df_stock.min())

    df_stock = df_stock.T.diff(axis=1).dropna(axis=1)
    print(df_stock.head())

    stocks, days = df_stock.shape
    print('days:', days, 'stocks:', stocks)

    #AUTOENCODER
    AE = ae.AutoEncoder(shape = days)
    conv_layers = {'conv1': {'filters': 4, 'kernel_size': 5, 'strides': 1},
                   #'conv2': {'filters': 2, 'kernel_size': 2, 'strides': 1},
                   #conv3': {'filters': 2, 'kernel_size': 2, 'strides': 1}
                   }
    pool_layers = {'pool1': {'pool_size': 5, 'strides': 5},
                   #'pool2': {'pool_size': 2, 'strides': 2},
                   #'pool3': {'pool_size': 2, 'strides': 2}
                   }
    kernels = AE.train(df_stock, conv_layers, pool_layers)
    df_kernels = pd.DataFrame(kernels.reshape(5,4))

    f, ax = st.create_axis()
    df_kernels.plot(ax=ax, legend=True, title='Kernels')
    path = os.path.join(FIGS, 'kernels.png')
    f.savefig(path, dpi=200, bbox_inches='tight')





if __name__ == '__main__':
    make_report()
