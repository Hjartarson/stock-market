import sys
import os
from datetime import datetime
now = datetime.now()

OMX30 = ['SKA-B','HM-B']
OMX30 = ['SKA-B','HM-B','NDA-SEK','ERIC-B','TELIA','ALIV-SDB','NCC-B','LUMI-SDB','THULE','BOL','GETI-B','AZN',
         'MTG-B', 'SAND','ALFA','ASSA-B','ATCO-B','INVE-B','KINV-B','SCA-B','TEL2-B','SKF-B']
OMX30 = ['ERIC-B']
#OMX30 = ['NDA-SEK']
#EXCHANGE = 'INDEXNASDAQ'
#STO = 'OMXS30'
EXCHANGE = 'STO'

from make_pred import MakePrediction
import os

def run_pred():
    mp = MakePrediction()
    fail_quote = []
    for quote in OMX30:
        print([quote, EXCHANGE, 'new'])
        last_date, pred = mp.make_pred([quote,EXCHANGE, 'new'])
        pred = pred.stack().round(2)
        with open(os.path.join(os.pardir,'pred.csv'), 'a') as f:
            d_str = quote+','+str(last_date)
            for p in pred.values:
                d_str = d_str+','+str(p)
            f.write(d_str)
            f.write('\n')
        # except:
        #     fail_quote = fail_quote + [quote]
        #     print('Could not process',quote)
        #     print("Unexpected error:", sys.exc_info()[0])
    # if len(fail_quote)>0:
    #     print('FAILED QUOTES')
    #     for qoute in fail_quote:
    #         print(qoute)
    # else:
    #     print('Success! All quotes obtained!')

def summarize_pred():
    print('summary')


if __name__ == '__main__':
    run_pred()