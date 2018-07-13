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

import style as st
st.set_style()

def make_report():
    print('making report')
    df_stock = pd.read_pickle(os.path.join(PROCESSED, 'df_cluster.pkl'), compression = 'gzip')
    print(df_stock.head())
    CLUSTERS = df_stock['cluster'].unique()
    print(CLUSTERS)

    for cluster in CLUSTERS:
        f, ax = st.create_axis()
        print('cluster:', cluster)
        df_cluster = df_stock[df_stock['cluster'] == cluster]
        columns = df_cluster.columns[df_cluster.columns != 'cluster']
        df_cluster = df_cluster[columns].T
        stocks = len(df_cluster.columns)
        print('nr stocks:', stocks)
        if df_cluster.empty:
            print('no stock in cluster')
        else:
            df_cluster.plot(ax=ax, legend=True, title='cluster:'+str(cluster))
            ax.axhline(0, ls = '--', color='r')
            st.set_month_xticks(df_cluster.index, ax)
            ax = st.fix_legend(ax, cols=6)
            path = os.path.join(FIGS, 'cluster_'+str(cluster)+'.png')
            f.savefig(path, dpi=200, bbox_inches='tight')



if __name__ == '__main__':
    make_report()
