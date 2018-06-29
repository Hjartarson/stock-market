import sys
import os
from datetime import datetime
now = datetime.now()

from make_pred import MakePrediction
import os

def get_quotes(what):
    if what == 'OMX30':
        EXCHANGE = 'STO'
        QUOTES = ['ALFA','ATCO-B','ALIV-SDB','AZN','ASSA-B',
                  'BOL',
                  'ERIC-B','ELUX-B',
                  'GETI-B',
                  'HM-B',
                  'INVE-B',
                  'KINV-B',
                  'LUMI-SDB','LUPE',
                  'MTG-B',
                  'NDA-SEK','NCC-B','NOKIA-SEK',
                  'SKA-B','SAND','SCA-B','SKF-B','SEB-C','SECU-B','SSAB-B','SWED-A','SWMA',
                  'TELIA','THULE','TEL2-B',
                  'VOLV-B']

    elif what == 'FOREX':
        EXCHANGE = 'CURRENCY'
        QUOTES = ['EURUSD','EURSEK','EURGBP']

    elif what == 'INDEX':
        EXCHANGE = 'INDEXNASDAQ'
        QUOTES = 'OMXS30'
    return QUOTES, EXCHANGE


def run_pred(what):
    mp = MakePrediction()
    QUOTES, EXCHANGE = get_quotes(what)
    fail_quote = []
    for quote in QUOTES:
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
    run_pred(sys.argv[1])