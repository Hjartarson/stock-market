
import numpy as np

class Strategies:
	
    #price
    price_open = 0
    price_max = 0
    price_min = np.inf
    price_init_max = 0
    price_init_min = np.inf
    
    #volume
    total_volume = 0
    
    #trades
    count_trades = 0
    
    def init_min_max_momentum(self, time, price, vol):
    
        self.total_volume += vol
        self.count_trades += 1
        
        price_max = max(self.price_max, price)
        price_min = min(self.price_min, price)
        
        #set opening price
        if self.price_open == 0:
            self.price_open = price
            
        #record initial period
        if time.hour == 9 & time.minute < 20:
            if self.price_init_max < price:
                print('new max', time, price)
                self.price_init_max = price
            if self.price_init_min > price:
                print('new min', time, price)
                self.price_init_min = price
                
        return 'hold'