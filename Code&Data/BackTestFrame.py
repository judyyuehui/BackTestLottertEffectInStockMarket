import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import scipy.io
from performance.py import performance
from backtest.py import backtest
import statsmodels.api as sm
import statsmodels.formula.api as smf

# IC
# self.factor return, real trading return
# IR
# max draw down
data = scipy.io.loadmat("CaseDataAll.mat")
Daily_Yield = pd.DataFrame(data['Daily_Yield'])
a_DataTrading = pd.DataFrame(data['a_DateTrading'])

a_Code = pd.DataFrame(data['a_Code'])

a_EOM = pd.DataFrame(data['a_EOM'])
a_EOM[0] = a_EOM[0].apply(lambda x: str(x))
a_EOM[0] = pd.to_datetime(a_EOM[0])
Mon_Yield = pd.DataFrame(data['Mon_Yield']).set_index(a_EOM[0])
#Mon_Yield = Mon_Yield.drop(['DateTime'], axis = 1)

Daily_ClosePriceAdj = pd.DataFrame(data['Daily_ClosePriceAdj'])
a_DataTrading = pd.DataFrame(data['a_DateTrading'])

a_Code = pd.DataFrame(data['a_Code'])



Daily_ClosePriceAdj.index = a_DataTrading[0]
Daily_ClosePriceAdj['DateTime'] = Daily_ClosePriceAdj.index
Daily_ClosePriceAdj['yearmonth'] = Daily_ClosePriceAdj['DateTime'].apply(lambda x: str(x)[:6])


def first_last(df):
    return df.iloc[[-1]]

Mon_Close = Daily_ClosePriceAdj.groupby('yearmonth').apply(first_last)
Mon_Close.index = pd.to_datetime(Mon_Close.DateTime.apply(lambda x: str(x)))

Mon_Close = Mon_Close.drop(['DateTime', 'yearmonth'], axis = 1)
Mon_Close.columns = a_Code[0]
Mon_Close = Mon_Close[:-1]

Daily_Yield.index = a_DataTrading[0]
Daily_Yield['DateTime'] = Daily_Yield.index
Daily_Yield['yearmonth'] = Daily_Yield['DateTime'].apply(lambda x: str(x)[:6])
MAX = Daily_Yield.groupby(Daily_Yield['yearmonth']).max()
MAX.index = pd.to_datetime(MAX.DateTime.apply(lambda x: str(x)))
MAX['DateTime'] = pd.to_datetime(MAX["DateTime"].apply(lambda x: str(x)[:6]+'01'))

MAX = MAX.drop(['DateTime'], axis = 1)
MAX.columns = a_Code[0]
MAX = MAX[:-1]

Mon_Yield.columns = MAX.columns
Mon_Close.columns = MAX.columns


yields = backtest(Mon_Yield, Mon_Close, MAX)
per = performance(yields)


rf = pd.read_csv('rf.csv')
rf['DateTime'] = pd.to_datetime(rf['trdmn'].apply(lambda x: str(x) + '01'))

def sharp(yields, rf):
    yields = pd.DataFrame(yields)
    yields = yields[yields.index.year>=2005]
    rf.index = yields.index
    yields['rf'] = rf['rf']
    return (yields['yields'] - yields['rf']).mean()/(yields['yields'] - yields['rf']).std()