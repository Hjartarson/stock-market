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
from os import listdir
from os.path import isfile, join
mypath = 'data/stocks'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles
```

```python

df = pd.DataFrame(columns = ['tick', 'datetime', 'volume', 'price'])

for file in onlyfiles:
    tick = file.split('.')[0]
    print(tick)
    df_tick = pd.read_csv('data\\stocks\\' + file)
    df_tick = df_tick.rename(columns = {'Local time':'datetime'})
    df_tick['tick'] = tick
    df_tick.columns = df_tick.columns.str.lower()
    df_tick['datetime'] = pd.to_datetime(df_tick['datetime'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc = True).dt.tz_convert(tz='Europe/Stockholm')
    df_tick = df_tick.sort_values(by='datetime')
    
    #FIX SPLIT
    if tick == 'INVE':
        df_tick.loc[df_tick['datetime'] < '2021-05-19', ['open','high','low','close']] = df_tick.loc[df_tick['datetime'] < '2021-05-19', ['open','high','low','close']]/4
        
    df_tick['price'] = df_tick[['close', 'open', 'low', 'high']].mean(axis=1)
    df_tick = df_tick.drop(columns=['high', 'low', 'open', 'close'])
    
    df = pd.concat([df, df_tick])

print(df.info())
df.head()
```

```python
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time
```

```python
df.set_index(['datetime', 'tick'])['price'].unstack().plot()
```

```python
df = df.set_index(['tick','date','time'])[['price']].unstack().ffill(axis=1).bfill(axis=1).stack()
df.head()
```

### CREATE FEATURES

```python
# ADD FEATURES

for minutes in [10, 20, 30]:
    print(minutes)
    df['feat_price_'+str(minutes)+'min_max'] = df.unstack()['price'].rolling(minutes, axis=1).max().stack()
    df['feat_price_'+str(minutes)+'min_max_shifted'] = df.unstack()['price'].rolling(minutes, axis=1).max().shift(minutes, axis=1).stack()
    df['feat_price_'+str(minutes)+'min_min'] = df.unstack()['price'].rolling(minutes, axis=1).min().stack()
    df['feat_price_'+str(minutes)+'min_min_shifted'] = df.unstack()['price'].rolling(minutes, axis=1).min().shift(minutes, axis=1).stack()
    
df.head()
```

### ADD RESPONS

```python
pred_time = 60
predict_up = 'resp_price_'+str(pred_time)+'min_max'
predict_down = 'resp_price_'+str(pred_time)+'min_min'

df[predict_up] = df.unstack()['price'].rolling(pred_time, axis=1).max().shift(-pred_time, axis=1).stack()
df[predict_down] = df.unstack()['price'].rolling(pred_time, axis=1).min().shift(-pred_time, axis=1).stack()
df.head()

df = df.reset_index()
```

### NORMALIZE

```python
features = [column for column in df.columns if 'feat_' in column]
df[features] = df[features].div(df['price'], axis=0)

respons = [column for column in df.columns if 'resp_' in column]
df[respons] = df[respons].div(df['price'], axis=0)

df.tail(5)
```

### MODEL

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
pct_change = 0.01

df[f'resp_price_{pred_time}min_max_up'] = 0
df.loc[df[f'resp_price_{pred_time}min_max']>1 + pct_change, f'resp_price_{pred_time}min_max_up'] = 1

df[f'resp_price_{pred_time}min_min_down'] = 0
df.loc[df[f'resp_price_{pred_time}min_min']<1 - pct_change, f'resp_price_{pred_time}min_min_down'] = 1

print(df[f'resp_price_{pred_time}min_max_up'].mean())
print(df[f'resp_price_{pred_time}min_min_down'].mean())
```

```python
from datetime import time

corona_low = '2020-03-01'
corona_high = '2020-05-01'

from_date = '2018-01-01'
to_date = '2022-01-01'


df_train = df[(df['date']>from_date) & (df['date']<to_date)].dropna(axis=0).copy()
df_test = df[((df['date']>=to_date))].dropna(axis=0).copy()

```

### BUY SIGNAL

```python
#MODEL
X = df_train.loc[:, features]
Y = df_train.loc[:, f'resp_price_{pred_time}min_max_up']

scale_pos_weight = Y[Y == 0].count()/Y[Y==1].count()
print('scale pos: ', scale_pos_weight)

model = XGBClassifier(#scale_pos_weight = scale_pos_weight, 
                      max_depth = 3, eval_metric="error",
                      n_jobs = 8, use_label_encoder=False)

model.fit(X, Y, eval_metric="error",
          verbose=False)
```

```python
X = df_test.loc[:, features]
Y = df_test.loc[:, f'resp_price_{pred_time}min_max_up']

y_pred = model.predict(X)

df_test[f'pred_price_{pred_time}min_max_up'] = y_pred
print(df_test[f'pred_price_{pred_time}min_max_up'].mean())
```

### SELL SIGNAL

```python
#MODEL
X = df_train.loc[:, features]
Y = df_train.loc[:, f'resp_price_{pred_time}min_min_down']

scale_pos_weight = Y[Y == 0].count()/Y[Y==1].count()
print('scale pos: ', scale_pos_weight)

model = XGBClassifier(scale_pos_weight = scale_pos_weight, 
                      max_depth = 3, eval_metric="error", missing=None,
                      n_jobs = 8, use_label_encoder=False)

model.fit(X, Y, eval_metric="error",
          verbose=False)
```

```python
X = df.loc[:, features]
Y = df.loc[:, f'resp_price_{pred_time}min_min_down']

y_pred = model.predict(X)

df[f'pred_price_{pred_time}min_min_down'] = y_pred
print(df[f'pred_price_{pred_time}min_min_down'].mean())
```

### EVALUATE

```python
sample = df_test[(df_test[f'pred_price_{pred_time}min_max_up']==1) & (df_test['date']>to_date) & (df_test['date']!='2022-05-02')].sample(1)[['tick', 'date']]

f, ax = plt.subplots(figsize= (12, 8))

tick = sample['tick'].values[0]
date = sample['date'].values[0]

print(tick, date)

df_cell = df[(df['tick']==tick) & (df['date']==date)].copy()
df_cell.set_index('time')['price'].plot(ax=ax, title=f'{tick} {date}')



df_cell = df_test[(df_test['tick']==tick) & (df_test['date']==date)].copy()
df_cell[f'pred_price_{pred_time}min_max_up'] = df_cell[f'pred_price_{pred_time}min_max_up'].replace(0,np.nan)
df_cell[f'pred_price_{pred_time}min_max_up'] = df_cell.groupby(['tick', 'date'])[f'pred_price_{pred_time}min_max_up'].ffill(60)

df_cell[df_cell[f'pred_price_{pred_time}min_max_up']==1].set_index('time')['price'].plot(ls='', marker='o', ax=ax, color='g')
#df_cell[df_cell[f'pred_price_{pred_time}min_min_down']==1].set_index('time')['price'].plot(ls='', marker='o', ax=ax, color='r')
```

```python
df[''] = df_test[f'pred_price_{pred_time}min_max_up']
```

```python
df_test
```

```python
df_cell[df_cell[f'pred_price_{pred_time}min_max_up']==1]
```
