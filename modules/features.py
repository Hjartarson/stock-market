import numpy as np

def add_features(df):
    # ADD MEASURES

    # SHIFT SINGLE
    # open/close.shift(1) = GAP
    # close/close.shift(1) = GAP + DAY CHANGE (WHAT YOU SEE)
    df = df.join(df['open'].div(df['close'].shift(periods=1))
                             .sub(1).mul(100).rename('x_close_open'))

    for days in np.arange(1,11):
        # look at at days back total change
        df = df.join(df['open'].div(df['open'].shift(periods=days))
                                 .sub(1).mul(100).rename('x_open_open_days_'+str(days)))
        # Look at days back daily change
        df = df.join(df['open'].shift(periods=days-1).div(df['open'].shift(periods=days))
                                 .sub(1).mul(100).rename('x_open_open_day_'+str(days)))

    for days in np.arange(1,11):
        # look at at days back total change
        df = df.join(df['high'].shift(periods=1).div(df['low'].shift(periods=days))
                                 .sub(1).mul(100).rename('x_low_high_days_'+str(days)))
        # Look at days back daily change
        df = df.join(df['high'].shift(periods=days).div(df['low'].shift(periods=days))
                                 .sub(1).mul(100).rename('x_low_high_day_'+str(days)))

    #PREDICTORS, SHIFT BOTH BACK IN TIME, SEE IN FUTURE
    #GAP FIRST DAY EXCLUDED SINCE CLOSE PRICE INCLUDED IN X (NO TIME TO TRADE)
    # close.shift(-1)/open.shift(1) = NEXT DAY OUTCOME
    # close.shift(-days)/open.shift(1) = NEXT X DAYS OUTCOME

    df = df.join(df['high'].div(df['low'])
                             .sub(1).mul(100).rename('y_low_high'))
    df = df.join(df['high'].div(df['open'])
                             .sub(1).mul(100).rename('y_open_high'))

    for days in np.arange(0,11):
        df = df.join(df['close'].shift(periods=-days).div(df['open'])
                                 .sub(1).mul(100).rename('y_open_close_days_'+str(days)))

    return df
