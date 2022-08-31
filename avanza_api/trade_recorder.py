import pandas as pd
import matplotlib.pyplot as plt


class TradeRecorder:

    df_trades = pd.DataFrame(columns = ['time', 'price', 'volume', 'action'])

    def __init__(self):
        print('Record Trades')
        
        
    def record_trade(self, time, price, volume, action):
        self.df_trades = self.df_trades.append({'time': time, 'price': price, 'volume': volume, 'action': action}, ignore_index=True)
        
    def save_trades(self):
        self.df_trades.to_pickle('trades.pkl', compression = 'gzip')
        
    def plot_trades(self):
        self.df_trades['handle'] = self.df_trades['price'].mul(self.df_trades['volume'])
        self.df_trades = self.df_trades.groupby('time')[['handle', 'volume']].sum()
        self.df_trades['price'] = self.df_trades['handle'].div(self.df_trades['volume'])

        f, ax = plt.subplots()
        self.df_trades['price'].plot(ax=ax)
        f.savefig('trades.png')