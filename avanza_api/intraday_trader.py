import pandas as pd
import datetime
import sys
import numpy as np
from signal import signal, SIGINT
from sys import exit
from time import time as timeit, ctime
from strategies import Strategies
from trade_recorder import TradeRecorder


class IntradayTrader:

    def __init__(self):
        """https://qluxzz.github.io/avanza/avanza.html#avanza.avanza.Avanza.subscribe_to_ids"""
        
        self.source_of_data = 'locals'
        #.channel = self.tick_id[channel]
        self.strategy = Strategies()
        self.traderecorder = TradeRecorder()
        
    def set_source(self, source):
        self.source_of_data = source
        
    def signal_handler(self, signal_received, frame):
        print('Exiting...')
        self.traderecorder.save_trades()
        self.traderecorder.plot_trades()
        sys.exit(0)

    def callback(self, data):
        if self.source_of_data == 'avanza_stream':
            time = datetime.datetime.strptime(ctime(data['data']['dealTime']/1000), "%a %b %d %H:%M:%S %Y")
        else:
            time = data['data']['time']
        price = data['data']['price']
        vol = data['data']['volume']
        string = f'time: {time}, price: {price}, vol: {vol}'
        print(string)
        
        action = self.strategy.init_min_max_momentum(time, price, vol)
        self.traderecorder.record_trade(time, price, vol, action)
                
        signal(SIGINT, self.signal_handler)
        
        

