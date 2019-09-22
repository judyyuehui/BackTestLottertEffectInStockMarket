import numpy as np
import pandas as pd
from time import time,localtime,strftime
import scipy.io
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt


class performance(object):
    def __init__(self, backtest, sep=False):
        self.yields = backtest.result['yields']
        self.returns = backtest.returns
        self.factor = backtest.factor
        self.sep = sep
        self.trade_date = backtest.trade_date_time
        self.year_index = list(zip(self.yields.index.to_series().resample('BAS').first(),
                                   self.yields.index.to_series().resample('BAS').last()))
        self.year = [val[0].to_pydatetime().year for val in self.year_index]

        max_drawdown = self.get_max_drawdown()
        self.max_drawdown = max_drawdown[0] if not sep else [val[0] for val in max_drawdown]
        self.max_drawdown_period = [str(max_drawdown[1]), str(max_drawdown[2])] if not sep else [
            [str(val[1]), str(val[2])] for val in max_drawdown]

        self.annualized_yield = self.get_annualized_yield()
        self.information_ratio = self.get_information_ratio()

        #self.information_coefficient = self.get_information_coefficient()

        self.output = (self.information_ratio, self.annualized_yield, self.max_drawdown)
        self.turnover = self.get_turnover()
        self.bpmg = self.get_bpmg()

    def get_turnover(self, delay=1):
        returns = self.returns.iloc[delay:]
        factor = self.factor.shift(delay).iloc[delay:]
        returns = returns.stack()
        factor = factor.stack()
        returns = [returns.loc[i] for i in returns.index]
        factor = [factor.loc[i] for i in factor.index]
        multi = pd.DataFrame([returns[i] * factor[i] for i in range(len(returns))])
        turnover = multi.abs().sum()/multi.abs().max()
        return turnover

    def get_bpmg(self):
        if not self.sep:
            series = self.yields
            if series.isnull().sum() != 0:
                raise ValueError('Empty Value Exists')
            return float(series.sum() / self.turnover)
        else:
            return [performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]]).get_bpmg() for i
                    in
                    range(len(self.year_index))]

    def get_max_drawdown(self):
        if not self.sep:
            series = self.yields.cumsum() + 1

            if series.isnull().sum() != 0:
                raise ValueError('Empty Value Exists')

            end_point = np.argmax(np.maximum.accumulate(series) - series)
            start_point = np.argmax(series[:end_point])
            num = - float(format(float(series[end_point] / series[start_point]) - 1, '0.4f'))
            return num, start_point.to_pydatetime(), end_point.to_pydatetime()
        else:
            return [performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]]).get_max_drawdown() for i
                    in
                    range(len(self.year_index))]

    def get_annualized_yield(self):
        if not self.sep:
            series = self.yields
            if not isinstance(series, pd.Series):
                raise TypeError("Only pd.Series is supported")
            return float(format(series.sum() / len(self.trade_date) * 250, '0.4f'))
        else:
            return [performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]]).get_annualized_yield() for
                    i in
                    range(len(self.year_index))]

    def get_information_ratio(self):
        series = self.yields
        if not self.sep:
            if not isinstance(series, pd.Series):
                raise TypeError("Only pd.Series is supported")
            return float(format(series.resample('d').sum()[self.trade_date].mean() /
                                series.resample('d').sum()[self.trade_date].std() *
                                math.sqrt(250), '0.4f'))
        else:
            return [performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]]).get_information_ratio()
                    for
                    i in range(len(self.year_index))]

    def get_information_coefficient(self, delay=1):
        returns = self.returns.iloc[delay:]
        factor = self.factor.shift(delay).iloc[delay:]
        returns = returns.stack()
        factor = factor.stack()
        returns = [returns.loc[i] for i in returns.index]
        factor = [factor.loc[i] for i in factor.index]

        return np.corrcoef(returns, factor)[1, 0]

    def yields_plot(self):
        series = self.yields.cumsum() + 1
        if not self.sep:
            if not isinstance(series, pd.Series):
                raise TypeError("Only pd.Series is supported")

            plt.figure(figsize=(10, 5))
            plt.grid(linestyle="--")
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.title("Accumulate Yields", fontsize=14, fontweight='bold')
            plt.xlabel("date-time", fontsize=13, fontweight='bold')
            plt.ylabel("yields", fontsize=13, fontweight='bold')
            series.plot()
            plt.show()

        else:
            return [performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]]).yields_plot() for
                    i in range(len(self.year_index))]

    def __str__(self):
        if not self.sep:
            return "Max Drawdown: {0}\nMax Drawdown Period: {1}\nAnnualized Yield: {2}\n" \
                   "Information Ratio: {3}\nInformation Coefficient: {4}\n".format(
                self.max_drawdown, self.max_drawdown_period, self.annualized_yield,
                self.information_ratio, self.information_coefficient)
        else:
            return '\n'.join(["Year: %s\n" % self.year[i] + str(
                performance(self.series.loc[self.year_index[i][0]:self.year_index[i][1]])) for i in
                              range(self.year.__len__())])

    __repr__ = __str__

