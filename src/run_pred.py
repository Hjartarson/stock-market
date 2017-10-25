

OMX30 = ['SKA-B','HM-B','NDA-SEK','ERIC-B','TELIA']
OMX30 = ['HM-B']
EXCHANGE = 'STO'

from make_pred import MakePrediction


if __name__ == '__main__':
    mp = MakePrediction()
    for quote in OMX30:
        print([quote, EXCHANGE, 'new'])
        mp.make_pred([quote,EXCHANGE, 'old'])