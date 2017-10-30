


import pandas as pd
from datetime import datetime, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import xgboost
import graphviz
import sys
now = datetime.now()
print(now)



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



class MakePrediction:

    def check_quality(self, df):
        # MUST FFILL, ELSE CHEATING
        return df.replace(0,np.nan).fillna(method='ffill')

    def get_data(self, get_new, info, Quote, interval_sec, period_length, period, exchange):
        if get_new == 'new':
            df = stocks.GoogleIntradayQuote(Quote, interval_sec, period_length, period, exchange)
            df.to_pickle(os.path.join(DATA,Quote+info+'.pkl'))
        else:
            df = pd.read_pickle(os.path.join(DATA,Quote+info+'.pkl'))
        return df


    def make_pred(self, argv):
        # OMXS30 Data
        # ['ALIV-SDB','NCC-B','LUMI-SDB','THULE','HM-B','VOLV-B','BOL','GETI-B','SKA-B','AZN']
        # Quotes = ['OMXS30']
        # exchange = 'INDEXNASDAQ'
        features = Features()
        visuals = Visuals()
        ml = MachineLearning()

        Quote = argv[0]
        exchange = argv[1]
        get_new = argv[2]
        # CREATE FOLDER
        OUT = os.path.join(FIG,Quote)
        if not os.path.exists(OUT):
            os.makedirs(OUT)
        # HIST DATA
        period_length = 6; period = 'Y'; interval_min = 60 * 24; interval_sec = 60 * interval_min
        df = self.get_data(get_new, '', Quote, interval_sec, period_length, period, exchange)

        df = df.groupby('datetime').sum()
        # CHECK QUALITY
        df = df.pipe(self.check_quality)

        # NEW DAY IF MARKET OPEN
        if now.time() > time(9,00):
            period_length = 1
            period = 'd'
            interval_min = 2
            interval_sec = 60 * interval_min
            df_last_day = self.get_data(get_new, '-TODAY', Quote, interval_sec, period_length, period, exchange)
            if df_last_day.empty:
                print('Could not get data')
            else:
                open_price = df_last_day[df_last_day['datetime'] == df_last_day['datetime'].min()]['open'].values[0]
                predict_day = df_last_day['datetime'].min().replace(hour=17, minute=30)
                if predict_day.date() != now.date():
                    print('Error! Todays data could not be obtained, using old data...')
                else:
                    # ADD NEW OPEN PRICE
                    dft = pd.DataFrame([[open_price, np.nan, np.nan, np.nan, np.nan]],
                                       columns=['open','high','low','close','volume'],
                                       index=[predict_day])
                    df = df.append(dft)
                    print('Index unique:',df.index.is_unique)

        from_date = str(df.index.min().date())
        to_date = str(df.index.max().date())
        days = np.busday_count(from_date, to_date) + 1
        print('ticks:\t', df.shape[0])
        print('from:\t', from_date)
        print('to:\t', to_date)
        print('days:\t', days)

        # ADD FEATURES
        df['x_volume'] = df['volume'].shift(1).copy()
        df = df.pipe(features.add_rolling)
        df.pipe(visuals.plot_data, Quote)
        x_var = [x for x in np.unique(df.columns) if 'x_' in x]
        df = df.pipe(feat.shift_columns, x_var, 10)
        x_var = [x for x in np.unique(df.columns) if 'x_' in x]

        # ADD OUTPUT
        df = df.pipe(feat.add_outcome, days=10)

        # TEST/PLOT PRED
        df.pipe(ml.xgb_pred_test, x_var).pipe(visuals.plot_pred,Quote)

        # MAKE PRED
        df = df.pipe(ml.xgb_pred, x_var)
        df.pipe(visuals.print_result, Quote)

        pred_var = [x for x in np.unique(df.columns) if 'pred_open_close_up_' in x]
        # f, ax = plt.subplots(1, 1)
        return to_date, df[pred_var].tail(1)


class Features:

    def add_rolling(self, df):
        for days_rolling in [3,5,10,20,30,40,50,60, 100]:
            df = df.join(df['open'].rolling(days_rolling).mean().rename('x_open_rolling_'+str(days_rolling)))
        for days_rolling in [10, 20]:
            df = df.join(df['x_volume'].rolling(days_rolling).mean().rename('x_volume_rolling_' + str(days_rolling)))
        return df

class MachineLearning:
    xgb = xgboost.XGBClassifier(max_depth=6, objective='binary:logistic')

    def xgb_pred_test(self, df,x_var):
        print('predicting days (testrun)...')
        for predict_days in np.arange(0, 11):
            daystr = "%02d" % (predict_days)
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
            self.xgb.fit(x_train, y_train)
            ypred = self.xgb.predict_proba(x_test)
            score = self.xgb.score(x_test, y_test)

            # Save prediction
            df = df.join(pd.DataFrame(ypred, columns=['pred_open_close_down_' + daystr,
                                                      'pred_open_close_up_' + daystr],
                                      index=test.index))
        return df

    def xgb_pred(self, df,x_var):
        print('predicting days...')

        for predict_days in np.arange(0, 11):
            daystr = "%02d" % (predict_days)
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

            self.xgb.fit(x_train, y_train)
            ypred = self.xgb.predict_proba(x_pred)

            df = df.join(pd.DataFrame(ypred[:,1], columns=['pred_open_close_up_' + daystr],
                                      index=df_pred.index))
        return df

class Visuals:

    def plot_pred(self, df,Quote):
        print('plot prediction...')
        conf_level = 0.9
        for days in np.arange(0, 11):
            daystr = "%02d" % (days)
            predict_variable = 'y_open_close_days_' + str(days) + '_up'

            df_buy = df[(df['pred_open_close_up_' + daystr] > conf_level)]
            df_sell = df[(df['pred_open_close_down_' + daystr] > conf_level)]

            agg = {predict_variable[:-3]: 'mean',
                   'pred_open_close_up_' + daystr: 'count'}
            buy_with_confidence = df_buy.groupby(predict_variable).agg(agg)
            agg = {predict_variable[:-3]: 'mean',
                   'pred_open_close_down_' + daystr: 'count'}
            sell_with_confidence = df_sell.groupby(predict_variable).agg(agg)
            trade_with_conf = buy_with_confidence.join(sell_with_confidence, lsuffix='_buy', rsuffix='_sell')
            trade_with_conf = trade_with_conf.round(1)
            f, ax = plt.subplots(1, 1)
            df['close'].plot(ax=ax, title=Quote)
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            try:
                df_buy['close'].plot(ls='', marker="o", ms=8, ax=ax, color='g')
                df_sell['close'].plot(ls='', marker="o", ms=8, ax=ax, color='r')
            except:
                print('No numeric data to plot\n')
            plt.table(cellText=trade_with_conf.values, colWidths=[0.25] * len(trade_with_conf.columns),
                      rowLabels=trade_with_conf.index,
                      colLabels=trade_with_conf.columns,
                      cellLoc='center', rowLoc='center',
                      loc='bottom')
            ax.grid()
            #ax.xlabel('')
            filepath = os.path.join(os.path.join(FIG,Quote),str(days)+'_days_pred.png')
            pl.save_fig(f, filepath)
            plt.close('all')
            #img = Image.open(filepath)
            #img.show()

    def plot_data(self, df, Quote):
        f, ax = plt.subplots(1, 1)
        ax_sec = ax.twinx()
        # for days_rolling in [3, 5, 10, 20, 30, 40, 50, 60]:
        #     df['open'].rolling(days_rolling).mean().plot(ax=ax,alpha=0.5)
        x_volume = [x for x in np.unique(df.columns) if 'x_volume_rolling' in x]
        x_open = [x for x in np.unique(df.columns) if 'x_open_rolling_' in x]
        df[x_open].plot(ax=ax, grid=False, title=Quote, legend=False)
        df[x_volume].plot(ax=ax_sec, grid=False, title=Quote, legend=False)
        ax.set_xlabel('')
        ax.set_yticks([])
        ax_sec.set_yticks([])
        filepath = os.path.join(os.path.join(FIG, Quote), '_raw.png')
        pl.save_fig(f, filepath)
        # img = Image.open(filepath)
        # img.show()

    def print_result(self, df, Quote):
        pred_var = [x for x in np.unique(df.columns) if 'pred_open_close_up_' in x]
        f, ax = plt.subplots(1, 1)
        df[pred_var].tail(11).round(2).pipe(pl.render_mpl_table,ax)
        filepath = os.path.join(os.path.join(FIG, Quote),'pred.png')
        pl.save_fig(f, filepath)




if __name__ == '__main__':
    mp = MakePrediction()
    mp.make_pred(sys.argv[1:])