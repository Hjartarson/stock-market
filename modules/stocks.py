import urllib.request as urlopen
import pandas as pd
import datetime
import numpy as np

def GoogleIntradayQuote(Quote,interval_seconds,period_length,period,exchange='STO'):
    df = pd.DataFrame(columns=['datetime','stock','open','high','low','close','volume'])

    url_string = "http://finance.google.com/finance/getprices?q={0}&x={1}".format(Quote,exchange)
    url_string += "&p={0}{1}&i={2}&f=d,o,h,l,c,v".format(period_length,period,interval_seconds)
    print(url_string)
    csv = urlopen.urlopen(url_string).readlines()
    csv = [x.decode("utf-8") for x in csv]
    for bar in range(7,len(csv)):
        if csv[bar].count(',')!=5:
            continue
        offset,close,high,low,open_,volume = csv[bar].split(',')
        if offset[0]=='a':
            day = float(offset[1:])
            offset = 0
        else:
            offset = float(offset)
        open,high,low,close,volume = [float(x) for x in [open_,high,low,close,volume]]
        dt = datetime.datetime.fromtimestamp(day+(interval_seconds*offset))
        df2 = pd.DataFrame([[dt,Quote,open,high,low,close,volume]],columns=['datetime','stock','open','high','low','close','volume'])
        df = df.append(df2,ignore_index = True)
    return df