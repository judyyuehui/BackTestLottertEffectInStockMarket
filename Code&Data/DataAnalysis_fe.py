import numpy as np
import pandas as pd
from time import time,localtime,strftime
import scipy.io
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

#加载数据
data = scipy.io.loadmat("CaseDataAll.mat")


#data processing
Daily_Yield = pd.DataFrame(data['Daily_Yield'])
a_DataTrading = pd.DataFrame(data['a_DateTrading'])

a_Code = pd.DataFrame(data['a_Code'])

#计算factor：MAX， MAX是每只股票每个月最大的日收益率，频率为月度数据
Daily_Yield.index = a_DataTrading[0]
Daily_Yield['DateTime'] = Daily_Yield.index
Daily_Yield['yearmonth'] = Daily_Yield['DateTime'].apply(lambda x: str(x)[:6])
MAX = Daily_Yield.groupby(Daily_Yield['yearmonth']).max()
MAX.index = pd.to_datetime(MAX.DateTime.apply(lambda x: str(x)))
MAX['DateTime'] = pd.to_datetime(MAX["DateTime"].apply(lambda x: str(x)[:6]+'01'))

#加载其他因子，包括EOM, BM, EP, EP, Illiq, IVol, SizeAll, SizeFlt, Turnover, Yield_1
Daily_TradingVolumn = pd.DataFrame(data['Daily_TradingVolumn'])
Daily_TradingVolumn.index = a_DataTrading[0]
Daily_TradingVolumn['yearmonth'] = Daily_TradingVolumn.index
Daily_TradingVolumn['yearmonth'] = Daily_TradingVolumn['yearmonth'].apply(lambda x: str(x)[:6])
volume = Daily_TradingVolumn.groupby(Daily_TradingVolumn['yearmonth']).sum()

a_EOM = pd.DataFrame(data['a_EOM'])
a_EOM[0] = a_EOM[0].apply(lambda x: str(x))
a_EOM[0] = pd.to_datetime(a_EOM[0])
Mon_BM = pd.DataFrame(data['Mon_BM']).set_index(a_EOM[0])
Mon_EP = pd.DataFrame(data['Mon_EP']).set_index(a_EOM[0])
Mon_Illiq = pd.DataFrame(data['Mon_Illiq']).set_index(a_EOM[0])
Mon_IVol = pd.DataFrame(data['Mon_IVol']).set_index(a_EOM[0])
Mon_SizeAll = pd.DataFrame(data['Mon_SizeAll']).set_index(a_EOM[0])
Mon_SizeFlt = pd.DataFrame(data['Mon_SizeFlt']).set_index(a_EOM[0])
Mon_Turnover = pd.DataFrame(data['Mon_TurnOver']).set_index(a_EOM[0])
Mon_Yield = pd.DataFrame(data['Mon_Yield']).set_index(a_EOM[0])
Mon_Yield_1 = pd.DataFrame(data['Mon_Yield_1']).set_index(a_EOM[0])


Mon_BM['DateTime'] = Mon_BM.index
Mon_BM['DateTime'] = Mon_BM['DateTime'].apply(lambda x: str(x)[:4] + str(x)[5:7])
Mon_BM['DateTime'] = pd.to_datetime(Mon_BM["DateTime"].apply(lambda x: str(x)[:6]+'01'))

Mon_EP['DateTime'] = Mon_BM['DateTime']
Mon_Illiq['DateTime'] = Mon_BM['DateTime']
Mon_IVol['DateTime'] = Mon_BM['DateTime']
Mon_SizeAll['DateTime'] = Mon_BM['DateTime']
Mon_SizeFlt['DateTime'] = Mon_BM['DateTime']
Mon_Turnover['DateTime'] = Mon_BM['DateTime']
Mon_Yield['DateTime'] = Mon_BM['DateTime']
Mon_Yield_1['DateTime'] = Mon_BM['DateTime']


#单因子排序

decile = pd.DataFrame(np.zeros((10,12)))
decile.columns = ['MAX', 'Mon_EP', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1','R_VW', 'decile']
decile.index = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
j = 0
start_date = pd.to_datetime('2005-01-01 00:00:00')
end_date = pd.to_datetime('2018-07-01 00:00:00')
na = 0
date = 1
for i in Mon_BM['DateTime']:
    if i <= start_date or i >= end_date:
        date = date + 1
        continue
    frame = MAX[MAX['DateTime'] == i]
    frame = frame.append(Mon_EP[Mon_EP['DateTime'] == i])
    frame = frame.append(Mon_BM[Mon_BM['DateTime'] == i])
    frame = frame.append(Mon_Illiq[Mon_Illiq['DateTime'] == i])
    frame = frame.append(Mon_IVol[Mon_IVol['DateTime'] == i])
    frame = frame.append(Mon_SizeAll[Mon_SizeAll['DateTime'] == i])
    frame = frame.append(Mon_SizeFlt[Mon_SizeFlt['DateTime'] == i])
    frame = frame.append(Mon_Turnover[Mon_Turnover['DateTime'] == i])
    frame = frame.append(Mon_Yield[Mon_Yield['DateTime'] == i])
    frame = frame.append(Mon_Yield_1[Mon_Yield_1['DateTime'] == i])

    frame.index = ['MAX', 'Mon_EP', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1']
    frame = frame.T[:-1]
    frame['Mon_Yield'] = frame['Mon_Yield']
    for name in frame.columns.tolist():
        frame[name] = frame[name].apply(lambda x: float(x))

    #frame['MAX'] = frame['MAX'].dropna()
    frame = frame.dropna()
    if frame.MAX.empty == True:
        na = na+1
        continue
    else:
        print(i)

    try:
        frame['decile'] = pd.qcut(frame['MAX'], 10, labels=False)
    except ValueError:
        continue

    frame['decile'] = frame['decile'].apply(lambda x: float(x))


    for name in frame.columns.tolist():
        if name != 'DateTime':
            frame[name] = frame[name].apply(lambda x: float(x))
    a = frame.groupby(frame['decile']).mean()
    SizeRatio = pd.DataFrame(frame.groupby(frame['decile'])['Mon_SizeAll'].sum())
    SizeRatio.columns = ['SizeAll']
    frame['SizeRatio'] = frame['decile'].apply(lambda x: SizeRatio.iloc(float(x))[0])
    frame['R_VW'] = frame['Mon_Yield'] * frame['Mon_SizeAll'] / frame['SizeRatio']
    a['R_VW'] = frame.groupby(frame['decile'])['R_VW'].mean()
    decile = decile + a
    j = j + 1
    #f_name = 'monthly_data/' + i.strftime('%Y%m%d') + '.csv'
    #frame.to_csv(f_name)

decile = decile / j


#双因子排序

def bivariate_sort(control_variable):
    decile = pd.DataFrame(np.zeros((10,10)))
    decile.columns = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]#columns是控制变量
    decile.index = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    j = 0
    start_date = pd.to_datetime('2005-01-01 00:00:00')
    end_date = pd.to_datetime('2018-07-01 00:00:00')

    for i in Mon_BM['DateTime']:
        if i <= start_date or i >= end_date:
            continue
        frame = MAX[MAX['DateTime'] == i]
        frame = frame.append(Mon_EP[Mon_EP['DateTime'] == i])
        frame = frame.append(Mon_BM[Mon_BM['DateTime'] == i])
        frame = frame.append(Mon_Illiq[Mon_Illiq['DateTime'] == i])
        frame = frame.append(Mon_IVol[Mon_IVol['DateTime'] == i])
        frame = frame.append(Mon_SizeAll[Mon_SizeAll['DateTime'] == i])
        frame = frame.append(Mon_SizeFlt[Mon_SizeFlt['DateTime'] == i])
        frame = frame.append(Mon_Turnover[Mon_Turnover['DateTime'] == i])
        frame = frame.append(Mon_Yield[Mon_Yield['DateTime'] == i])
        frame = frame.append(Mon_Yield_1[Mon_Yield_1['DateTime'] == i])
        frame.index = ['MAX', 'Mon_EP','Mon_BM' ,'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1']
        frame = frame.T[:-1]

        frame.index = a_Code[0]

        for name in frame.columns.tolist():
            frame[name] = frame[name].apply(lambda x: float(x))
        frame['R_VW'] = frame['Mon_Yield'] * frame['Mon_SizeAll']
        frame['R_VW'] = frame['R_VW'] / frame['Mon_SizeAll'].sum()
        #frame['MAX'] = frame['MAX'].dropna()
        frame = frame.dropna()
        if frame.MAX.empty == True:
            na = na+1
            continue
        else:
            print(i)

        try:
            frame['decile_B'] = pd.qcut(frame[control_variable], 10, labels=False)
        except ValueError:
            continue

        for k in range(0, 10):
            frame_temp = frame[frame['decile_B'] == k]
            frame_temp['count'] = 1
            try:
                frame_temp['decile_M'] = pd.qcut(frame_temp['MAX'], 10, labels=False)
            except ValueError:
                continue

            a = frame_temp.groupby(frame_temp['decile_M'])['Mon_Yield'].sum()
            b = frame_temp.groupby(frame_temp['decile_M'])['count'].sum()
            c = a / b

            decile[k] = decile[k] + c


        j = j + 1
        #f_name = 'monthly_data/' + i.strftime('%Y%m%d') + '.csv'
        #frame.to_csv(f_name)

    decile = decile / j
    decile['aver'] = 0
    for i in range(0, 10):
        decile['aver'] = decile['aver'] + decile[i]
    decile['aver'] = decile['aver'] / 10

    return decile


decile = bivariate_sort('Mon_BM')

decile.to_csv('bi_sort_Mon_EP.csv')




#three_factor_model计算alpha,beta

three_factor_model = pd.read_csv('three_factor_model.csv')
three_factor_model['DateTime'] = pd.to_datetime(three_factor_model['Mon'].apply(lambda x: str(x) + '01'))
rf = pd.read_csv('rf.csv')
rf['DateTime'] = pd.to_datetime(rf['trdmn'].apply(lambda x: str(x) + '01'))

alpha = []
beta = []
t_value_con = []
t_value_Mkt = []
start_date = pd.to_datetime('2005-01-01 00:00:00')
end_date = pd.to_datetime('2018-07-01 00:00:00')
for n in range(0, 10):
    regression = pd.DataFrame(np.zeros((1, 5)))
    regression.columns = ['Mkt_Rf', 'SMB', 'VMG', 'yields', 'rf']
    for i in Mon_BM['DateTime']:
        if i <= start_date or i >= end_date:
            continue
        frame = MAX[MAX['DateTime'] == i]
        frame = frame.append(Mon_Yield[Mon_Yield['DateTime'] == i])

        frame.index = ['MAX', 'yields']
        frame = frame.T[:-1]
        frame['yields'] = frame['yields'].shift(-1)
        frame = frame.dropna()
        if frame.MAX.empty == True:
            na = na + 1
            continue
        else:
            print(i)

        frame['count'] = 1
        try:
            frame['decile'] = pd.qcut(frame['MAX'], 10, labels=False)
        except ValueError:
            continue
        a = frame.groupby(frame['decile'])['yields'].sum()
        b = frame.groupby(frame['decile'])['count'].sum()
        c = a/b

        three = three_factor_model[three_factor_model['DateTime'] == i]
        rf_temp = rf[rf['DateTime'] == i]
        regression.loc[i] = [float(three['Mkt_Rf']),float(three['SMB']),float(three['VMG']), float(a[n]), float(rf_temp['rf'])]


    regression['y'] = regression['yields'] - regression['rf']
    regression = regression[1:]
    X = regression[['Mkt_Rf', 'SMB', 'VMG']]
    X = sm.add_constant(X)
    model = sm.OLS(regression['y'], X)
    results = model.fit()
    alpha.append(results.params['const'])
    beta.append(results.params['Mkt_Rf'])
    t_value_con.append(results.tvalues['const'])
    t_value_Mkt.append(results.tvalues['Mkt_Rf'])

model_fit = pd.DataFrame([alpha, t_value_con, beta, t_value_Mkt])
model_fit.index = ['alpha', 'alpha_t', 'beta', 'beta_t' ]
model_fit = model_fit.T

model_fit.to_csv('2005-2018/alpha_beta.csv')



#FM regression

j = 0
start_date = pd.to_datetime('2005-01-01 00:00:00')
end_date = pd.to_datetime('2018-07-01 00:00:00')
na = 0
date = 1
fm_regression = pd.DataFrame(np.zeros((1, 12)))
fm_regression.columns = ['MAX', 'Mon_EP', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll',
       'Mon_SizeFlt', 'Mon_Turnover', 'Mon_Yield', 'Mon_Yield_1', 'stkcd',
       'DateTime']
for i in Mon_BM['DateTime']:
    if i <= start_date or i >= end_date:
        date = date + 1
        continue
    frame = MAX[MAX['DateTime'] == i]
    frame = frame.append(Mon_EP[Mon_EP['DateTime'] == i])
    frame = frame.append(Mon_BM[Mon_BM['DateTime'] == i])
    frame = frame.append(Mon_Illiq[Mon_Illiq['DateTime'] == i])
    frame = frame.append(Mon_IVol[Mon_IVol['DateTime'] == i])
    frame = frame.append(Mon_SizeAll[Mon_SizeAll['DateTime'] == i])
    frame = frame.append(Mon_SizeFlt[Mon_SizeFlt['DateTime'] == i])
    frame = frame.append(Mon_Turnover[Mon_Turnover['DateTime'] == i])
    frame = frame.append(Mon_Yield[Mon_Yield['DateTime'] == i])
    frame = frame.append(Mon_Yield_1[Mon_Yield_1['DateTime'] == i])

    frame.index = ['MAX', 'Mon_EP', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1']

    frame = frame.T[:-1]

    frame['stkcd'] = a_Code[0]
    frame['DateTime'] = i
    frame['Mon_Yield'] = frame['Mon_Yield'].shift(-1)
    frame['Mon_Yield_1'] = frame['Mon_Yield_1'].shift(-1)
    for name in frame.columns.tolist():
        if name != 'DateTime':
            frame[name] = frame[name].apply(lambda x: float(x))
    frame['R_VW'] = frame['Mon_Yield'] * frame['Mon_SizeAll']
    frame['R_VW'] = frame['R_VW'] / frame['Mon_SizeAll'].sum()
    #frame['MAX'] = frame['MAX'].dropna()
    frame = frame.dropna()
    fm_regression = fm_regression.append(frame)
fm_regression = fm_regression[1:]
fm_regression.to_csv('fm_regression.csv')


def ols_coef(x,formula):
    return smf.ols(formula,data=x).fit().params


fm_regression.dropna()
for name in fm_regression.columns.tolist():
    if name != 'DateTime':
        fm_regression[name] = fm_regression[name].apply(lambda x: float(x))


def fm_summary(p):
    s = p.describe().T
    s['std_error'] = s['std'] / np.sqrt(s['count'])
    s['tstat'] = s['mean'] / s['std_error']
    return s[['mean', 'std_error', 'tstat']]

s = pd.DataFrame(np.zeros((1, 3)))
s.columns = ['mean', 'std_error', 'tstat']
formular = 'Mon_Yield ~ 1 + MAX + '
for n in ['Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                       'Mon_Yield_1']:
    formular = formular + n + ' + '
    reg1 = (fm_regression.groupby('DateTime').apply(ols_coef, 'Mon_Yield ~ 1 + MAX +' + n))
    s = s.append(fm_summary(reg1))

formular = formular[:-2]
reg1 = (fm_regression.groupby('DateTime').apply(ols_coef, formular))
overall_s = fm_summary(reg1)

#所有数据的统计量分析

def description(a):
    des = pd.DataFrame(np.zeros((1, 6)))
    des.columns = ['mean', 'median', 'var', 'std', 'skew', 'kurt']
    des['mean'][0] = a.mean()
    des['median'][0] = a.median()
    des['var'][0] = a.var()
    des['std'][0] = a.std()
    des['skew'][0] = a.skew()
    des['kurt'][0] = a.kurt()
    return des

descrip = pd.DataFrame(np.zeros((1, 6)))
descrip.columns = ['mean', 'median', 'var', 'std', 'skew', 'kurt']
for n in ['MAX', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1']:
    descrip = descrip.append(description(fm_regression[n]))

descrip = descrip[1:]
descrip.index = ['MAX', 'Mon_BM', 'Mon_Illiq', 'Mon_IVol', 'Mon_SizeAll', 'Mon_SizeFlt', 'Mon_Turnover',
                      'Mon_Yield', 'Mon_Yield_1']

descrip.to_csv('description.csv')

#Time-Series plot of MAX
max = fm_regression.groupby(['DateTime'])['MAX'].mean()

plt.figure(figsize=(10, 5))
plt.grid(linestyle="--")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.title("Time Series Data of MAX (average)", fontsize=14, fontweight='bold')
plt.xlabel("date-time", fontsize=13, fontweight='bold')
plt.ylabel("average maximum daily yield", fontsize=13, fontweight='bold')
plt.show()
plt.plot(max)

