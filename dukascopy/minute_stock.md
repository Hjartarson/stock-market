---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib as mpl

import os
import pandas as pd

import matplotlib.pyplot as plt
```

```python
# https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed

file = 'INVEB.SESEK_Candlestick_1_M_BID_01.08.2019-12.07.2022.csv'

#file = 'DEU.IDXEUR_Candlestick_1_M_BID_01.08.2019-16.07.2022.csv'
#file = 'HMB.SESEK_Candlestick_1_M_BID_01.08.2019-13.07.2022.csv'

#file = 'XAUUSD_Candlestick_1_M_BID_01.08.2019-09.07.2022.csv'
#file = 'NDA.SESEK_Candlestick_1_M_BID_01.08.2019-16.07.2022.csv'
#file = 'ERICB.SESEK_Candlestick_1_M_BID_01.08.2019-23.07.2022.csv'
#file = 'SEBA.SESEK_Candlestick_1_M_BID_01.08.2019-23.07.2022.csv'
#file = 'TLSN.SESEK_Candlestick_1_M_BID_01.08.2019-23.07.2022.csv'

df = pd.read_csv('data\\' + file)
df = df.rename(columns = {'Local time':'datetime'})
df.columns = df.columns.str.lower()
df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc = True).dt.tz_convert(tz='Europe/Stockholm')

#FIX SPLIT
if file[:5] == 'INVEB':
    df.loc[df['datetime'] < '2021-05-19', ['open','high','low','close']] = df.loc[df['datetime'] < '2021-05-19', ['open','high','low','close']]/4

df = df.sort_values(by='datetime')

# STRIP
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time
df['price'] = df[['close', 'open', 'low', 'high']].mean(axis=1)
df = df.drop(columns=['high', 'low', 'open', 'close', 'datetime'])
df.head()
```

```python
df_price = df.set_index(['date','time'])[['price']].unstack().ffill(axis=1).bfill(axis=1).stack().reset_index()
df_vol = df.set_index(['date','time'])[['volume']].unstack().ffill(axis=1).bfill(axis=1).stack().reset_index()
df = df_price.merge(df_vol , how='right', on=['date', 'time'])

df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour

df['datetime_diff'] = df['datetime'].diff()
df['hour_gap'] = df['datetime_diff'] / np.timedelta64(1, 'h')
df['new_day'] = 0
df.loc[df['hour_gap'] > 10,'new_day'] = 1
df['return'] = df['price'].pct_change()
df['return'] = df['return'].add(1)
df.head()
```

```python
df.set_index(['datetime'])['return'].cumprod().plot()
```

```python
df.head()
```

```python
#df = df.set_index(['date','time'])
#df['return_cumday'] = df['price'].unstack().pct_change(axis = 1).add(1).cumprod(axis=1).stack()
#df['price_max_day'] = df['price'].unstack().cummax(axis = 1).stack()
    

for minutes in [20, 60, 120, 240, 480]:
    #df['return_'+str(minutes)+'min'] = df['price'].pct_change(minutes).add(1)
    #df['return_'+str(minutes)+'min_shifted'] = df['return_'+str(minutes)+'min']
    
    #df['return_'+str(minutes)+'min_avg'] = df['return'].rolling(minutes).mean()
    #df['return_'+str(minutes)+'min_median'] = df['return'].rolling(minutes).median()
    #df['feat_return_'+str(minutes)+'min_std'] = df['return'].rolling(minutes).std()
    
    
    #df['feat_price_'+str(minutes)+'min'] = df['price'].rolling(minutes).mean()
    df['feat_price_'+str(minutes)+'min_max'] = df['price'].rolling(minutes).max()
    df['feat_price_'+str(minutes)+'min_max_shifted'] = df['feat_price_'+str(minutes)+'min_max'].shift(minutes)
    df['feat_price_'+str(minutes)+'min_min'] = df['price'].rolling(minutes).min()
    df['feat_price_'+str(minutes)+'min_min_shifted'] = df['feat_price_'+str(minutes)+'min_min'].shift(minutes)
    
#df = df.reset_index()

features = [column for column in df.columns if 'feat_' in column]
df[features] = df[features].div(df['price'], axis=0)

df.tail(5)
```

```python
#df.loc[df['new_day']==1, 'feat_return_new_day'] = df['return']
#df['feat_return_new_day'] = df['feat_return_new_day'].ffill()
#features = [column for column in df.columns if 'feat_' in column]
#df.tail()
```

## DO ML

```python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics 
from sklearn import tree
```

```python
hold_time_h = 1
pct_change = 0.005

df['price_pct'] = df['price'].shift(-hold_time_h*60).rolling(hold_time_h*60).max().div(df['price'])
df['price_pct_up'] = 0
df.loc[df['price_pct']>1 + pct_change, 'price_pct_up'] = 1

df['price_pct'] = df['price'].shift(-hold_time_h*60).rolling(hold_time_h*60).min().div(df['price'])
df['price_pct_down'] = 0
df.loc[df['price_pct']<1 - pct_change, 'price_pct_down'] = 1

print(df['price_pct_up'].mean())
print(df['price_pct_down'].mean())
```

```python
from datetime import time

corona_low = '2020-03-01'
corona_high = '2020-05-01'
from_date = '2018-01-01'
to_date = '2022-01-01'
df_train = df[((df['date']<corona_low) | (df['date']>corona_high))  & 
              ((df['date']>from_date) & (df['date']<to_date))
            & ((df['time']>time(8, 0)) & (df['time']<time(16, 30)))
             ].copy()
df_test = df[((df['date']>=to_date))].copy()
```

```python
#MODEL
X = df_train.loc[:, features]
Y = df_train.loc[:, 'price_pct_up']

scale_pos_weight = Y[Y == 0].count()/Y[Y==1].count()
print('scale pos: ',scale_pos_weight)

model = XGBClassifier(#scale_pos_weight = scale_pos_weight, 
                      max_depth = 3, eval_metric="error",
                      n_jobs = 8, use_label_encoder=False)

model.fit(X, Y, eval_metric="error",
          #eval_set=eval_set,
          verbose=False)
```

```python
X = df.loc[:, features]
Y = df.loc[:, 'price_pct_up']

y_pred = model.predict_proba(X)
#accuracy = accuracy_score(Y, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

df['pred_price_pct_up'] = y_pred[:,1]
print(df['pred_price_pct_up'].mean())
```

### SELL

```python
X = df_train.loc[:, features]
Y = df_train.loc[:, 'price_pct_down']

model = XGBClassifier(#scale_pos_weight = scale_pos_weight, 
                      max_depth = 3, eval_metric="error",
                      n_jobs = 8, use_label_encoder=False)

model.fit(X, Y, eval_metric="error",
          #eval_set=eval_set,
          verbose=False)
```

```python
X = df.loc[:, features]
Y = df.loc[:, 'price_pct_down']

y_pred = model.predict_proba(X)
#accuracy = accuracy_score(Y, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

df['pred_price_pct_down'] = y_pred[:,1]
print(df['pred_price_pct_down'].mean())
```

# MAKE MODELS

```python
df['action'] = np.nan
df['price_buy'] = np.nan

buy_strategy = 'ml'
sell_strategy = 'ml'

signal = 'avg'

# BUY
if buy_strategy == 'ml':
    #df['action'] = df['pred_price_pct_up'] - df['pred_price_pct_down']
    df.loc[df['pred_price_pct_up']>0.5, 'action'] = 1

if sell_strategy == 'ml':
    #df['action'] = df['pred_price_pct_up'] - df['pred_price_pct_down']
    df.loc[df['pred_price_pct_down']>0.5, 'action'] = -1
    
#df['action'] = df.groupby('date')['action'].ffill()
    
if buy_strategy == 'blank':
    pass
    
    
# Momentum
if buy_strategy == 'momentum':
    df.loc[#(df['return_new_day'].sub(1).abs()>=0.005) &# (df['return_new_day']>1) &
           (df['price_30min_min']>df['price_60min_min']) &
           (df[f'return_5min_{signal}']>df[f'return_10min_{signal}'])
           & (df[f'feat_return_30min_std']>df[f'feat_return_60min_std'])
           & (df[f'feat_return_15min_std']>df[f'feat_return_30min_std'])
           & (df[f'return_10min_{signal}']>df[f'return_15min_{signal}'])
           & (df[f'return_15min_{signal}']>df[f'return_30min_{signal}'])
           #& (df['return_20min_avg']>df['return_30min_avg'])
           #& (df['price']==df['price_60min_max'])
           #& (df['return_10min_std']>df['return_20min_std'])
           #& (df['return_cumday_30min_avg']>df['return_cumday_60min_avg'])
           #& (df['price'] >= df['price_max_day']) &
           #& (df['return_30min_median']>1)
           , 'action'] = 1


# Pullback
if buy_strategy == 'pullback':
    df.loc[(df['return_5min']>df['return_10min']) & 
           (df['return_10min']<df['return_15min']) & 
           (df['return_15min']<df['return_20min']) & 
           (df['return_20min']<df['return_30min']), 'action'] = 1
    


df.loc[df['action']==1, 'price_buy'] = df['price']
df['price_buy'] = df.groupby('date')['price_buy'].ffill()
    
    
if sell_strategy == 'time':
    df['action'] = df.groupby('date')['action'].ffill(limit=2*60)
    #df['action'] = df['action'].fillna(-1)
    
if sell_strategy == 'profit':
    df.loc[(df['price'] > df['price_buy']*1.01), 'action'] = -1
    df['action'] = df.groupby('date')['action'].ffill()
    
# SELL
if sell_strategy == 'momentum':
    df.loc[#(df['price']==df['feat_price_60min_min']) &
           (df[f'feat_price_5min']<df[f'feat_price_10min']) &
           (df[f'feat_price_10min']<df[f'feat_price_15min'])
           & (df[f'feat_price_15min']<df[f'feat_price_30min'])
        & (df[f'feat_price_30min']<df[f'feat_price_60min'])
        & (df[f'feat_price_60min']<df[f'feat_price_120min'])
           #& (df[f'return_30min_{signal}']<df[f'return_60min_{signal}'])
           , 'action'] = -1

#df = df.set_index(['date','time'])
#df['price_max'] = df['fika_price'].unstack().cummax(axis=1).stack()
#df = df.reset_index()

#df['fika_sell'] = np.nan
#df.loc[(df['fika_price'] < df['fika_max']*0.995),'fika_sell'] = 1

# HOLD STOCK UNTIL SELL SIGNAL OR DAY END
#if sell_strategy not in ['blank', 'ml']:
#df['action'] = df.groupby('date')['action'].ffill(3*60)
#df['action'] = df.groupby('date')['action'].shift(1)

```

```python
#df['action'] = df['action'].replace(-1,np.nan)
f, ax = plt.subplots(figsize= (12, 8))
df_cell = df[(df['date']>'2022-01-01')]
df_cell.set_index('datetime')['return'].cumprod().plot(ax=ax)
df_cell.loc[df['action']==1].set_index('datetime')['return'].sub(1).cumsum().add(1).plot(ax=ax)
df_cell.loc[df['action']==1].set_index('datetime')['return'].cumprod().plot(ax=ax)
ax.legend(['buy_hold', 'buy_buy', 'buy_invest'])
#df[df['datetime'].isin()]
```

```python
df['action_number'] = df['action'].replace(-1,0).fillna(0).diff().replace(-1, 0).cumsum()
df_actions = df[(df['action']==1) & (df['year']==2022)].groupby(['action_number'])['return'].agg(('prod', 'count'))
df_actions.sort_values(by='prod')
df_actions['count'].shape[0]/df['date'].nunique()
```

```python
print((df_actions['prod'].mean()-1)*100)
df_actions['prod'].plot(kind='hist', bins=100)
```

```python
(df_actions['prod'].mean()-1)*100#*100000*10
#df[df['action']==1].groupby('date')['return'].prod().sort_values(ascending=False)
```

```python
dates = df[(df['action']==1) & (df['date']>='2022-04-01') & (df['date']<'2022-07-01')]['date'].sort_values().unique()
print(len(dates))
#random_date = np.random.choice(dates, 10, replace = False)
#random_date = df[df['new_day']==0].groupby('date')['return'].prod().sort_values().tail(20).index
#random_date = df.groupby('date')[['price']].agg(['min', 'max']).pct_change(axis =1).sort_values(by=('price', 'max')).tail(20).index
#random_date = df[df['action']==1].groupby('date')['return'].prod().sort_values(ascending=False).index


for day in dates[0:20]:
    f, ax = plt.subplots(figsize= (12, 8))
    
    hour = 8
    
    df_filt = df[(df['date']==day) & (df['hour']>hour)]

    df_cum = df_filt.set_index('time')['return'].cumprod()
    df_cum.plot(title=f'{day}'[:10],ax=ax, lw=3, alpha = 1)
    #df[(df['date']==day)].set_index('time')['return_10min_avg'].plot(ax=ax, lw=2)
    #df[(df['date']==day)].set_index('time')['return_20min_avg'].plot(ax=ax, lw=2)
    #df[(df['date']==day)].set_index('time')['return_30min_avg'].plot(ax=ax, lw=2)
    
    
    df_cum.loc[df[(df['date']==day) & (df['hour']>hour) & (df['action']==1)]['time'].values].plot(ax=ax, ls='', marker='o', ms=8, color='orange', alpha=1)
    df_cum.loc[df[(df['date']==day) & (df['hour']>hour) & (df['action']==-1)]['time'].values].plot(ax=ax, ls='', marker='o', color='r', alpha=1)
    
    ax2 = ax.twinx()
    df_filt.set_index('time')['pred_price_pct_up'].plot(ax=ax2, color ='r', alpha = 0.6)
    #df[(df['action']==1) & (df['date']==day) & (df['hour']>hour)].set_index('time')['return'].cumprod().plot(ax=ax2)
    #df_cum.loc[df[df['return_10min']==1.002]['datetime'].values].plot(ax=ax, ls='', marker='o')
    ax.axvline('09:00', color='grey', ls='--')
    ax.axvline('10:00', color='grey', ls='--')
    #ax2.axvline('11:00', color='grey', ls='--')
    #ax2.axhline(1, color='grey', ls='--')
    
```

```python
f, ax = plt.subplots(figsize= (12, 8))

dates = df['date'].unique()
random_date = np.random.choice(dates, 10, replace = False)
df_cell = df[df['date'].isin(random_date)].set_index(['date','time'])['price'].unstack(level=0)

df_cell.div(df_cell.iloc[0,:]).plot(ax=ax, legend = False)
```

```python


df['price_fft'] = df['price'].apply(np.fft.fft)

df['price_fft'].plot()
```

```python
avg_daily = df.set_index(['date', 'time'])['return_cumday'].sub(1).unstack(level=0).dropna()

avg_daily_fft  = avg_daily.apply(np.fft.rfft).mean(axis=1)#.apply(np.fft.irfft)
avg_daily_fft_std  = avg_daily.apply(np.fft.rfft).std(axis =1)

min_range = np.arange(10,240,1) 
min_components = avg_daily_fft.values[min_range]
min_components_std = avg_daily_fft_std.values[min_range]

f, ax = plt.subplots(figsize= (12, 8))
ax.plot(min_range, abs(min_components))
#ax.plot(min_range, min_components+min_components_std)
#ax.plot(min_range, min_components-min_components_std)
```

```python
f, ax = plt.subplots(figsize= (12, 8))

minute = 30

avg_daily_rol = avg_daily.apply(np.fft.rfft).iloc[minute, :].rolling(20).mean()
avg_daily_rol_std = avg_daily.apply(np.fft.rfft).iloc[minute, :].rolling(20).std()

avg_daily_rol.abs().plot(ax=ax)
#avg_daily_rol.add(avg_daily_rol_std).plot(ax=ax)
#avg_daily_rol.sub(avg_daily_rol_std).plot(ax=ax)
```

```python
avg_daily.apply(np.fft.rfft)
```

```python
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq

signal = df['return_cumday'].sub(1).dropna()

W = fftfreq(signal.size, d=1)

f_signal = rfft(signal)
cut_f_signal = f_signal.copy()
cut_f_signal[(W>0.001)] = 0

cut_signal = irfft(cut_f_signal)

f, ax = plt.subplots(figsize= (12, 8))
ax.plot(f_signal)
#ax2 = ax.twinx()
#ax2.plot(cut_signal, ls='--', color='red')

W
```

```python
df['return_cumday'].sub(1)#.plot()
```

```python
W.max()
```

```python


time   = np.linspace(0,10,2000)
signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)

W = fftfreq(signal.size, d=signal.index[1]-signal.index[0])

W = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)

# If our original signal time was in seconds, this is now in Hz    
cut_f_signal = f_signal.copy()
cut_f_signal[(W<6)] = 0

cut_signal = irfft(f_signal)
```

#### BUY LOGIC

```python
from datetime import time

df['fika_entry'] = df['price'].shift(45)
df['fika_entry'] = df['price'].div(df['fika_entry'])

df.loc[(df['time']!=time(10, 0)), 'fika_entry'] = np.nan
df.loc[(df['time']==time(10, 0)) & (df['fika_entry']<=1), 'fika_entry'] = np.nan
df.loc[(df['time']==time(10, 0)) & (df['fika_entry']>1), 'fika_entry'] = df['price']

#df['fika_entry'] = df['fika_entry'].shift(1)

df = df.set_index(['date','time'])
df['fika_entry'] = df['fika_entry'].unstack().ffill(axis=1).shift(1, axis=1).stack()
df = df.reset_index()

df.loc[~df['fika_entry'].isna(),'fika_price'] = df['price']
df.loc[~df['fika_entry'].isna(),'fika_buy'] = 1

df.head(70).tail(20)
```

```python
df = df.set_index(['date','time'])
df['fika_max'] = df['fika_price'].unstack().cummax(axis=1).stack()
df = df.reset_index()

df.head(70).tail(20)
```

```python
# SELL LOGIC, 
df['fika_sell'] = np.nan
df.loc[(df['fika_price'] < df['fika_max']*0.995),'fika_sell'] = 1

df = df.set_index(['date','time'])
df['fika_sell'] = df['fika_sell'].unstack().ffill(axis=1).stack()
df = df.reset_index()
```

```python
df['fika_hold'] = np.nan

df.loc[(df['fika_buy']==1) & (df['fika_sell'].isna()), 'fika_hold'] = 1
df['fika_hold'] = df['fika_hold']#.shift(1)
df.head(70).tail(20)
```

```python
df.loc[df['fika_hold']==1].set_index('datetime')['return'].cumprod()
```

```python
f, ax = plt.subplots(figsize= (12, 8))
df.set_index('datetime')['return'].cumprod().plot(ax=ax)
df.loc[df['fika_hold']==1].set_index('datetime')['return'].cumprod().plot(ax=ax)
#df[df['datetime'].isin()]
```

```python
dates = df[df['fika_hold']==1]['date'].unique()
df.loc[(df['date'].isin(dates))].groupby('date')['return'].prod().sort_values().tail(10)
```

```python
dates = df[df['fika_hold']==1]['date'].unique()
random_date = df.loc[(df['date'].isin(dates))].groupby('date')['return'].prod().sort_values().tail(10).index
random_date = np.random.choice(dates, 20)

for day in random_date[:10]:
    random_date = day
    print(df[(df['date']==random_date) & (df['new_day']==1)]['return'])
    #gap = df[(df['date']==random_date) & (df['new_day']==1)].iloc[0]['return']
    f, ax = plt.subplots(figsize= (12, 8))
        
    df_cum = df[(df['date']==random_date)].set_index('time')['return'].cumprod()
    df_cum.plot(title=str(random_date),ax=ax, lw=3)
    df_cum.loc[df[(df['date']==random_date) & (df['fika_hold']==1)]['time'].values].plot(ax=ax, ls='', marker='o')
    #df_cum.loc[df[df['return_10min']==1.002]['datetime'].values].plot(ax=ax, ls='', marker='o')
    ax2 = ax.twinx()

    #df[(df['date']==random_date) & (df['return_10min']==1.002)].set_index('time')['return'].cumprod().plot(ax=ax, color='r', ls='', marker='o')
    #df[df['date']==random_date].set_index('time')['pct_change'].rolling(10).agg(lambda x : x.prod()).plot(ax=ax2, color='orange')
    #df[(df['date']==random_date)].set_index('time')['volume'].rolling(10).mean().plot(ax=ax2, color='r')
    #ax2.set(zorder=99)
    #ax2.axhline(0, color='grey', ls='--')
    ax.axvline('10:00', color='grey', ls='--')
    #ax2.axvline('11:00', color='grey', ls='--')
    ax.axvline('09:15', color='grey', ls='--')
```

```python
df.groupby('new_day')['return'].prod()
```

```python
df[(df['date']==random_date)].iloc[0]['pct_change']
```

```python

random_date = np.random.choice(df['date'].unique(), 10)

for day in random_date:
    random_date = day
    gap = df[(df['date']==random_date) & (df['new_day']==1)].iloc[0]['return']
    f, ax = plt.subplots(figsize= (12, 8))
    df[(df['date']==random_date) & (df['new_day']==0)].set_index('time')['return'].add(1).cumprod().plot(title=str(random_date)+' | '+str(gap),ax=ax, lw=3)
    ax2 = ax.twinx()

    df[df['date']==random_date].set_index('time')['return_60min'].plot(ax=ax2, color='r')
    #df[df['date']==random_date].set_index('time')['pct_change'].rolling(10).agg(lambda x : x.prod()).plot(ax=ax2, color='orange')
    #df[(df['date']==random_date)].set_index('time')['volume'].rolling(10).mean().plot(ax=ax2, color='r')
    #ax2.set(zorder=99)
    ax2.axhline(0, color='grey', ls='--')
    ax2.axvline('10:00', color='grey', ls='--')
    ax2.axvline('11:00', color='grey', ls='--')
    ax2.axvline('09:15', color='grey', ls='--')
```

```python
df[df['date']==random_date].set_index('time')['pct_change'].rolling(10).mean()
```

```python
df[df['date']==random_date].set_index('time')['pct_change']
```

```python
df_cell = df[df['year']==2022].set_index(['month', 'day','time'])['close'].unstack(level=[0,1]).bfill().ffill()
df_cell = df_cell.div(df_cell.iloc[0], axis=1)
df_cell
```

```python
f, ax = plt.subplots(figsize= (12, 8))
df_cell.sample(1, axis=1).plot(ax=ax, legend=False)
```

```python

```

```python
df.groupby(['datetime'])['pct_change'].prod().cumprod().plot()
```

```python
f, ax = plt.subplots(figsize= (20, 10))

df.groupby(['date', 'new_day'])['pct_change'].prod().unstack().fillna(1).cumprod().plot(ax=ax)
df.groupby(['date'])['pct_change'].prod().cumprod().plot(ax=ax)
```

```python
df['pct_change'].prod()
```

```python
df.groupby('new_day')['pct_change'].prod()
```

```python
f, ax = plt.subplots(figsize= (20, 10))
df_cell = df[df['year']==2022].groupby(['month', 'day', 'time'])['close'].mean().unstack(level=[0,1])
df_cell.div(df_cell.iloc[0, :], axis=1).sort_values(by=df_cell.index[0], axis=1)#.plot(ax=ax, legend=False)
```
