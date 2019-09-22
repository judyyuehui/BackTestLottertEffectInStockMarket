ReadMe:

数据分析部分:DataAnalysis.py:

按照说明数据加载、单因子排序、双因子排序、三因子模型回归、FM回归、数据统计量分析、时间序列作图这些部分
可更改数据加载时间
用到了CaseData.mat，three_factor_model的数据（后者转化为csv格式）


回测系统：
1. BackTest.py:
class object back_test
输入return, closeprice, factor三张表（横轴为公司序列，纵轴为时间timestamp）
得到每个交易时刻进行交易的公司序号（例子见trading_candidate.csv）
得到交易的yields

2. Performance.py
class object performance
输入back_test的数据
得到交易策略的表现结果（InformationRatio, Maxdrawdown, turnover, annualize yields）
通过yield_plot可以画出accumulated yield的图（格式、标题已设置好）

3. BackTestFrame.py
加载回测所需数据（调整格式等）
调用BackTets, Performance等文件得到回测结果
定义sharp ratio（由于需要risk free rate的数值，因此定义在这里而非performance.py）