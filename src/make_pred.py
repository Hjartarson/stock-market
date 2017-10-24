


import pandas as pd
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import xgboost
import sys
today = str(datetime.datetime.now().date())
print(today)



CURR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT =  os.path.join(CURR_PATH, os.pardir)
DATA =  os.path.join(ROOT, 'data')
FIG =  os.path.join(ROOT, 'fig')
sys.path.append(os.path.join(ROOT,'modules'))

import stocks as stocks
import features as feat
import plot as pl
import style as style
style.set_style()

def add_rolling(df):
    for days_rolling in [3,5,10,20,30,40,50,60]:
        df = df.join(df['open'].rolling(days_rolling).mean().rename('x_open_rolling_'+str(days_rolling)))
    return df

def xgb_pred_test(df,x_var):
    print('predicting days...')
    xgb = xgboost.XGBClassifier(max_depth=20, objective='binary:logistic')

    for predict_days in np.arange(0, 11):
        print(predict_days)
        predict_variable = 'y_open_close_days_'+str(predict_days)+'_up'
        df_pred = df[x_var+[predict_variable]].copy()
        df_pred = df_pred.dropna()

        df_pred['is_train'] = np.random.uniform(0, 1, len(df_pred)) <= .75
        train, test = df_pred[df_pred['is_train'] == True].copy(), df_pred[df_pred['is_train'] == False].copy()
        x_train = train[x_var].values
        x_test = test[x_var].values

        y_train = train[predict_variable].values
        y_test = test[predict_variable].values

        # xgboost
        xgb.fit(x_train, y_train)
        ypred = xgb.predict_proba(x_test)
        score = xgb.score(x_test, y_test)

        # Save prediction
        df = df.join(pd.DataFrame(ypred, columns=['pred_open_close_down_' + str(predict_days),
                                                  'pred_open_close_up_' + str(predict_days)],
                                  index=test.index))
    return df

def xgb_pred(df,x_var):
    print('predicting days...')
    xgb = xgboost.XGBClassifier(max_depth=20, objective='binary:logistic')

    for predict_days in np.arange(0, 11):
        predict_variable = 'y_open_close_days_'+str(predict_days)+'_up'
        print(predict_days)
        df_train = df[[predict_variable]+x_var].copy().dropna()
        df_pred = df[[predict_variable]+x_var].tail(predict_days+1)



    #     df_pred = df[x_var+[predict_variable]].copy()
    #     #df_pred = df_pred.dropna()
    #
        x_train = df_train[x_var].values
        y_train = df_train[predict_variable].values
        x_pred = df_pred[x_var].values
    #
    #     # xgboost
        xgb.fit(x_train, y_train)
        ypred = xgb.predict_proba(x_pred)
    #
    #     # Save prediction
        df = df.join(pd.DataFrame(ypred[:,1], columns=['pred_open_close_up_' + str(predict_days)],
                                  index=df_pred.index))
    return df

def plot_pred(df,Quote):
    print('plot prediction...')
    conf_level = 0.9
    days = 10
    predict_variable = 'y_open_close_days_' + str(days) + '_up'

    df_buy = df[(df['pred_open_close_up_' + str(days)] > conf_level)]
    df_sell = df[(df['pred_open_close_down_' + str(days)] > conf_level)]

    agg = {predict_variable[:-3]: 'mean',
           'pred_open_close_up_' + str(days): 'count'}
    buy_with_confidence = df_buy.groupby(predict_variable).agg(agg)
    agg = {predict_variable[:-3]: 'mean',
           'pred_open_close_down_' + str(days): 'count'}
    sell_with_confidence = df_sell.groupby(predict_variable).agg(agg)
    trade_with_conf = buy_with_confidence.join(sell_with_confidence, lsuffix='_buy', rsuffix='_sell')
    print(trade_with_conf.T)
    f, ax = plt.subplots(1, 1)
    df['close'].plot(ax=ax, title=Quote)

    df_buy['close'].plot(ls='', marker="o", ms=8, ax=ax, color='g')
    df_sell['close'].plot(ls='', marker="o", ms=8, ax=ax, color='r')
    ax.grid()
    filepath = os.path.join(FIG,Quote+'-PRED.png')
    pl.save_fig(f, filepath)

    img = Image.open(filepath)
    img.show()

def print_result(df):
    for days in np.arange(0,11):
        output_var = 'y_open_close_days_' + str(days) + '_up'
        pred_var = 'pred_open_close_up_' + str(days)
        print(df[[output_var, pred_var]].tail(11))

def check_quality(df):
    # MUST FFILL, ELSE CHEATING
    print(df.replace(0,np.nan).fillna(method='ffill').info())
    return df.replace(0,np.nan).fillna(method='ffill')

def plot_raw_data(df, Quote):
    f, ax = plt.subplots(1, 1)
    df.plot(ax = ax, secondary_y = 'volume', grid = True, title = Quote)
    filepath = os.path.join(FIG, Quote + '-RAW.png')
    pl.save_fig(f, filepath)
    img = Image.open(filepath)
    img.show()

def make_pred(argv):
    # OMXS30 Data
    # ['ALIV-SDB','NCC-B','LUMI-SDB','THULE','HM-B','VOLV-B','BOL','GETI-B','SKA-B','AZN']
    # Quotes = ['OMXS30']
    # exchange = 'INDEXNASDAQ'
    exchange = 'STO'
    Quote = argv[0]
    get_new = argv[1]

    # HIST DATA
    period_length = 6
    period = 'Y'
    interval_min = 60 * 24
    interval_sec = 60 * interval_min
    if get_new == 'new':
        df = stocks.GoogleIntradayQuote(Quote, interval_sec, period_length, period, exchange)
        df.to_pickle(os.path.join(DATA,Quote+'.pkl'))
    else:
        df = pd.read_pickle(os.path.join(DATA,Quote+'.pkl'))
    from_date = str(df['datetime'].min().date())
    to_date = str(df['datetime'].max().date())
    days = np.busday_count(from_date, to_date) + 1
    print('ticks:', df.shape[0])
    print('from:', from_date)
    print('to:', to_date)
    print('days:', days)

    # NEW DAY
    period_length = 1
    period = 'd'
    interval_min = 2
    interval_sec = 60 * interval_min
    if get_new == 'new':
        df_last_day = stocks.GoogleIntradayQuote(Quote, interval_sec, period_length, period, exchange)
        df_last_day.to_pickle(os.path.join(DATA,Quote+'-TODAY.pkl'))
    else:
        df_last_day = pd.read_pickle(os.path.join(DATA, Quote+'-TODAY.pkl'))
    open_price = df_last_day[df_last_day['datetime']==df_last_day['datetime'].min()]['open'].values[0]
    predict_day = df_last_day['datetime'].min().replace(hour=17,minute=30)
    print('Predict day:',predict_day)


    df = df.groupby('datetime').sum()
    # CHECK QUALITY
    df = df.pipe(check_quality)
    df.pipe(plot_raw_data, Quote)

    # ADD NEW OPEN PRICE
    dft = pd.DataFrame([[open_price, np.nan, np.nan, np.nan, np.nan]],
                       columns=['open','high','low','close','volume'],
                       index=[predict_day])
    df = df.append(dft)






    # ADD FEATURES
    df = df.pipe(add_rolling)
    # ADD OUTPUT
    df = df.pipe(feat.add_output_variables, days=10)
    df['x_volume'] = df['volume']
    x_var = [x for x in np.unique(df.columns) if 'x_' in x]
    print(x_var)

    # TEST PRED
    df_test = df.pipe(xgb_pred_test, x_var)
    df_test.pipe(plot_pred,Quote)
    y_var = [x for x in np.unique(df.columns) if 'y_' in x]
    pred_var = [x for x in np.unique(df.columns) if 'pred_' in x]
    # MAKE PRED
    df = df.pipe(xgb_pred, x_var)
    #df.pipe(print_result)
    # PLOT PRED




if __name__ == '__main__':
    make_pred(sys.argv[1:])