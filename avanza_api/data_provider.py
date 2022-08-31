import asyncio
from avanza import Avanza, ChannelType
from intraday_trader import IntradayTrader
from time import time as timeit, ctime
import pandas as pd
import datetime
import sys
import numpy as np

class DataProvider:

    source_of_data = 'none'
    
    
    tick_id = {
    'LEO':'643616',
    'EVO':'549768',
    'KIND':'56267'
    }
    
    def __init__(self, channel):
        """https://qluxzz.github.io/avanza/avanza.html#avanza.avanza.Avanza.subscribe_to_ids"""
        print('initiate')
        
        self.intraday_trader = IntradayTrader()
        self.channel = self.tick_id[channel]
        
      
    
       
    async def subscribe_to_channel(self, avanza: Avanza):
        await avanza.subscribe_to_id(
            ChannelType.TRADES,
            self.channel,
            self.intraday_trader.callback
        )

    def go_live(self):
        self.intraday_trader.set_source('avanza_stream')
        avanza = Avanza({
        'username': 'HjartarsonErik',
        'password': 'uzLfiSqA',
        'totpSecret': 'HRN33P7MFHRHY6VH5HQHR54RGEH6NODQ'
        })
        
        asyncio.get_event_loop().run_until_complete(
            self.subscribe_to_channel(avanza)
        )
        asyncio.get_event_loop().run_forever()

    def go_local(self, file):
        self.intraday_trader.set_source('local')
        df = pd.read_csv(file, skiprows=1, delimiter=';', usecols = ['Execution Time', 'Price', 'Volume'])
        print(df.info())
        df['time'] = pd.to_datetime(df['Execution Time'])
        df = df.sort_values(by='time')
        
        counter = 0
        for idx, row in df.iterrows():
            start = timeit()
            data = {'data':{'time':row['time'], 'price':row['Price'], 'volume':row['Volume']}}
            self.intraday_trader.callback(data)
               
            if counter%1000 == 0:
                print("total time taken this loop: ", timeit() - start)
            counter += 1

    

trade = DataProvider('EVO')
trade.go_live()
#trade.go_local('evo_20210427.csv')
#trade = IntradayTraderBacktest('evo_20210427.csv')