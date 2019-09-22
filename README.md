# Back Test Lottery Effect In Stock Market
Calculate Lottery Effect in American Market &amp; Build a simple backtest system to see factor performances

The trading strategy constructed in a relative term, that is, we long the stock with the factor value within top x% of whole data and short the last 5%, construct a zero-investment portfolio 

# File Descriptions:
## Report
    analysis the extreme return factor, explain the possible logic base on prospect theory, present its performance analysis  



## Code&Data
    A. Simple Back Test System: 
    three files in total. use the BackTestFrame.py to got the combined result   
      a. BackTest.py:    
          define class object back_test  
          initial: return, closeprice, factor三张表（横轴为公司序列，纵轴为时间timestamp）  
          functions: i. get trading candidates(the stock number)(see trading_candidate.csv to get a sample output）  
                    ii. get the return yields   
      b. Performance.py  
          define class object performance  
          initial: the data from BackTest.py(a backtest variable)  
          returns: 1. get data performance (InformationRatio, Maxdrawdown, turnover, annualize yields）  
                   2. (by choice) use the function yield_plot to get the plot of accumulated yield (***already set up the title&format***)  
                
      c. BackTestFrame.py  
        load the data used in backtest（make adjustments to the format of the data）  
        use functions in BackTest.py & Performance.py to get the backtest result  

        also, we define sharp ratio in this part（we didn't included it in the Performance.py because we need the data of risk free rate）  
        
    B. Output & Performance of the factor：   
         The related data of comps' financials  
         fm_regression.csv: factor data needed in fama-macbeth regression   
         trading_candidate(sample): the trading candidates data from backtest.py   
         
 ## NumericalResult
     analysize the data in statistcal methods，save the performance in .csv  
     since the performace result deviated in two periods (2005-2015 & 2015-2018), so we divided the data into 3 files: 2005-2018， 2005-2015， 2015-2018, record the result in each file  
     a. simple_sort.csv：decile sorting (one variable)  
     b. Mon_BM, Mon_EP, Mon_Illiq, Mon_IVol, Mon_SizeAll: bivarite sortings  
     c. alpha_beta.csv: alpha, beta & t values from fama-macbeth returns  
     d. fm_regression_result：fama-macbeth regression results  
     
 ## Graphs
     a. MAX_ts: MAX（The final factor we choice）time-serious curve of its value  
     b. Factor%i_tradingRatio=%n: the backtest accumulated yield curve at different trading rate   
     
