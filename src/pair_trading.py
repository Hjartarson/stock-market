import pandas as pd
from datetime import datetime, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

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



df = stocks.GoogleIntradayQuote('HM-B',8*60*60,1,'Y')


print(df.head())

f, ax = df.plot()