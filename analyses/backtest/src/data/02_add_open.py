
import os
import sys
import pandas as pd
import numpy as np
import datetime

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(CURR_PATH, os.pardir, os.pardir)
ROOT =  os.path.join(PROJ_ROOT, os.path.pardir, os.path.pardir)
DATA_RAW = os.path.join(PROJ_ROOT, "data","raw")
DATA_IM = os.path.join(PROJ_ROOT, "data","intermediate")
sys.path.append(os.path.join(ROOT,'modules'))

def add_open(quote, open, interval, period):
    file = quote+'-'+interval+'-'+period+'.pkl'
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    today_open = pd.to_datetime(today + ' 09:00:00')
    print('loading file:',file)
    df = pd.read_pickle(os.path.join(DATA_RAW, file))
    df_open = {'stock':quote,
               'open':float(open),
               'high':np.nan,
               'low':np.nan,
               'close':np.nan,
               'volume':np.nan}
    df_open = pd.DataFrame(df_open, index = [today_open])
    df = df.append(df_open)
    print(df.tail())
    print(df.info())

    df.to_pickle(os.path.join(DATA_RAW, quote + '-' + interval + '-' + period + '.pkl'))

if __name__ == '__main__':
    quote = sys.argv[1]
    open = sys.argv[2]
    if len(sys.argv) == 3:
        interval = '1d'
        period = '1Y'
    else:
        interval = sys.argv[3]
        period = sys.argv[4]
    add_open(quote, open, interval, period)