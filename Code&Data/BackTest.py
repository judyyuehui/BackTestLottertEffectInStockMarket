import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import scipy.io
from performance.py import performance
import statsmodels.api as sm
import statsmodels.formula.api as smf


class backtest(object):
    """
    单因子回测

    """

    def __init__(self, returns, close, factor, freq='1month', trade_ratio=0.05):

        self.close = close
        self.returns = returns
        self.factor = factor

        self.trade_ratio = trade_ratio

        # 检查数据完整性
        if not all(self.returns.index == self.factor.index):
            print("trade datetime does not match, thus exit")
            exit()

        if not all(self.returns.columns == self.factor.columns):
            print("universe does not match, thus exit")
            exit()

        self.universe = list(set(self.returns.columns).intersection(set(self.factor.columns)))
        self.stock_numb = factor.shape[1]

        # 生成交易时间
        self.trade_date_time = self.factor.index.tolist()
        #self.trade_year = sorted(list(set(map(dt.datetime.year, self.trade_date_time))))

        # 生成交易数量
        self.trade_number = int(math.floor(self.trade_ratio * self.stock_numb))
        self.result = self.get_yields()

    def get_candicate(self):
        """
        选择factor score头部和尾部的column index作为trade candicate
        sample output见coding&data部分trading_candidate(sample).csv
        """

        trade_candicate = self.factor.apply(lambda series: series.fillna(series.mean()).argsort(), axis=1)

        trade_candicate = trade_candicate.iloc[:,
                          list(range(self.trade_number)) + list(range(len(self.factor.columns)))[-self.trade_number:]]

        trade_candicate.columns = range(self.trade_number * 2)

        trade_candicate = trade_candicate.astype(int)

        trade_candicate = trade_candicate.apply(lambda series: series.apply(lambda i: self.universe[i]))

        return trade_candicate

    def get_yields(self):
        """
        # 选出t时刻score最高最低两个品种所对应的t+1时刻的return
        """

        trade_candicate = self.get_candicate()

        # 取出trade_candicate在下一个时间周期上的return
        yields_strategy = trade_candicate.shift(1).apply(
            lambda series: self.returns.loc[series.name].loc[series].values if series.isnull().sum() == 0 else [
                                                                                                                 np.nan] * self.trade_number * 2,
            axis=1)

        yields_strategy = pd.DataFrame.from_items(zip(yields_strategy.index, yields_strategy.values)).T

        # short self.factor score low part
        yields_strategy_short = - yields_strategy.iloc[:, :self.trade_number].fillna(0)
        # long self.factor score high part
        yields_strategy_long = yields_strategy.iloc[:, self.trade_number:].fillna(0)

        # combine long and short portfolio
        yields_strategy = (yields_strategy_short.mean(axis=1) + yields_strategy_long.mean(axis=1)) / 2.0

        # 获取交易成本
        #cost = self.get_cost(trade_candicate)
        #cost_fee = 0
        #cost_slip = cost['slip']

        result = pd.DataFrame({'yields': yields_strategy})

        return result
