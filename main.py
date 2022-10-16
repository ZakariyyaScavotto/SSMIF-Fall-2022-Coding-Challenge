# Zakariyya Scavotto Fall 2022 SSMIF Coding Challenge

# Import libraries for data calculations
import numpy as np
import pandas as pd
import scipy as sp
# Import pandas_datareader.data for reading in stock and equity data
import pandas_datareader.data as web


def generateDataframe():
    '''Creates a pandas DataFrame of the adjusted close stock history for 2021
    for Coca-Cola (KO), Tesla (TSLA) and the SPDR S&P500 Trust ETF (SPY)'''
    # Creates a DataFrame for each stock and ETF by getting the data for the year range
    koDF = web.DataReader('KO', 'yahoo', start='2021-01-01',
                          end='2021-12-31')['Adj Close']
    tslaDF = web.DataReader(
        'TSLA', 'yahoo', start='2021-01-01', end='2021-12-31')['Adj Close']
    spyDF = web.DataReader(
        'SPY', 'yahoo', start='2021-01-01', end='2021-12-31')['Adj Close']
    # Combines the 3 DataFrames into one and renames the Adj Close columns to make clear which columns corresponds to which stock/fund
    combinedDF = pd.concat([koDF, tslaDF], axis=1, join='inner')
    combinedDF = pd.concat([combinedDF, spyDF], axis=1, join='inner')
    combinedDF.columns = ['KO Adj Close', 'TSLA Adj Close', 'SPY Adj Close']
    return combinedDF


def statistics(koData, tslaData):
    '''Computes statistical test on the two equities to see if there is a statistically
    significant difference in the mean daily returns or the volatilities'''
    pass


'''References for learning financial terms and concepts for metrics method: 
https://www.investopedia.com/terms/v/volatility.asp
https://www.investopedia.com/articles/04/092904.asp
https://www.investopedia.com/terms/s/sharperatio.asp
https://www.investopedia.com/terms/d/downside-deviation.asp
https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
'''


def metrics(koData):
    '''Computes the volatility, 95% VaR, Sharpe Ratio, Downside Deviation
    and Maximum Drawdown for Coca-Cola (KO)'''
    pass


'''References for learning financial terms and concepts for capm method:
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
https://www.investopedia.com/terms/a/alpha.asp
'''


def capm(tslaData):
    '''Calculates the Beta and Alpha for Tesla (TSLA)'''
    pass


def main():
    '''Method to run code'''
    # Generate the DataFrame with adjusted closing price history data
    closeHistoryDF = generateDataframe()
    print('Finished generating DataFrame')
    # print(closeHistoryDF.head())
    # Execute the 3 methods
    statistics(closeHistoryDF.loc[:, "KO Adj Close"],
               closeHistoryDF.loc[:, "TSLA Adj Close"])
    metrics(closeHistoryDF.loc[:, "KO Adj Close"])
    capm(closeHistoryDF.loc[:, "TSLA Adj Close"])


if __name__ == '__main__':
    main()
