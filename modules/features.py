import numpy as np

def add_input_variables(df,days=10):
    # TODAY YOU ONLY KNOW OPEN!!
    # Everything has to be shifted (positive shifted) except open.

    # open/close.shift(1) = GAP
    df = df.join(df['open'].div(df['close'].shift(periods=1))
                             .sub(1).mul(100).rename('x_close_open'))

    for day in np.arange(1,days+1):
        # look at days back total change
        df = df.join(df['open'].div(df['open'].shift(periods=day))
                                 .sub(1).mul(100).rename('x_open_open_days_'+str(day)))
        # Look at days back daily change
        df = df.join(df['open'].shift(periods=day-1).div(df['open'].shift(periods=day))
                                 .sub(1).mul(100).rename('x_open_open_day_'+str(day)))
        # LOW to HIGH
        df = df.join(df['high'].shift(periods=1).div(df['low'].shift(periods=day))
                                 .sub(1).mul(100).rename('x_low_high_days_'+str(day)))
        # Look at days back daily change
        df = df.join(df['high'].shift(periods=day).div(df['low'].shift(periods=day))
                                 .sub(1).mul(100).rename('x_low_high_day_'+str(day)))
    return df

def shift_columns(df, columns, days=10):
    for day in np.arange(1, days + 1):
        new_col_nammes = [s + '_'+str(day) for s in columns]
        df_new = df[columns].shift(periods=day)
        df_new.columns = new_col_nammes
        df = df.join(df_new)
    return df


def add_outcome(df, days=10):
    #PREDICTORS, SHIFT BOTH BACK IN TIME, SEE IN FUTURE
    #GAP FIRST DAY EXCLUDED SINCE CLOSE PRICE INCLUDED IN X (NO TIME TO TRADE)
    # close.shift(-1)/open.shift(1) = NEXT DAY OUTCOME
    # close.shift(-days)/open.shift(1) = NEXT X DAYS OUTCOME
    df = df.join(df['high'].div(df['low'])
                             .sub(1).mul(100).rename('y_low_high'))
    df = df.join(df['high'].div(df['open'])
                             .sub(1).mul(100).rename('y_open_high'))

    for day in np.arange(0,days+1):
        df = df.join(df['close'].shift(periods=-day).div(df['open'])
                                 .sub(1).mul(100).rename('y_open_close_days_'+str(day)))
        df = df.join(df['y_open_close_days_'+str(day)].pipe(to_bin))

    return df


def to_bin(df):
    # ADD CAT INSTEAD OF FLOAT
    name = df.name
    df = df.to_frame()
    df[name+'_up'] = df[name]
    mask = (df[name] < 0)
    df.loc[df[name][mask].index, name+'_up'] = 0
    mask = df[name] > 0
    df.loc[df[name][mask].index, name+'_up'] = 1
    return df[name+'_up']

def up_down_streak(df):
    up_days = 0
    down_days = 0
    name = df.name
    df = df.to_frame()
    df[name+'_streak'] = df[name]
    for idx, value in df[name+'_streak'].iteritems():
        if value==False:
            down_days = down_days-1
            up_days = 0
            df.set_value(idx,name+'_streak',down_days)
        elif value==True:
            up_days = up_days+1
            down_days = 0
            df.set_value(idx,name+'_streak',up_days)
        else:
            df.set_value(idx,name+'_streak',0)
            down_days = 0
            up_days = 0
    return df[name+'_streak']