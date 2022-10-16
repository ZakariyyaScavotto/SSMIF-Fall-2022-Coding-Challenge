# Zakariyya Scavotto Fall 2022 SSMIF Coding Challenge

# Import listed libraries from instructions for data calculations and visualization
import numpy as np
import pandas as pd
import scipy as sp
# Import pandas_datareader.data for reading in stock and equity data
import pandas_datareader.data as web
# Import default math library to help with calculations
import math


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
    # Combine the 3 DataFrames into one, rename columns to make clear which column corresponds to which stock/fund
    combinedDF = pd.concat([koDF, tslaDF], axis=1, join='inner')
    combinedDF = pd.concat([combinedDF, spyDF], axis=1, join='inner')
    combinedDF.columns = ['KO Adj Close', 'TSLA Adj Close', 'SPY Adj Close']
    return combinedDF


def statistics(koData, tslaData):
    '''Computes statistical test on the two equities (KO and TSLA) to see if there is a statistically
    significant difference in the mean daily returns or the volatilities'''
    '''
    Check in Dad's orange book, but f or z test? 
    WRITE OUT ASSUMPTIONS
    Assume both equities are normally distributed, different variances
    Comparing volatilites/returns between the two equities
    Visualization of stock prices should help with picturing
    Trick: take coke stock price on 1/4, subtract every price throughout year by 1/4, multiply  by 100, indexing to be based on 100
        do same for TSLA, can then graph both series over the year and see what happens
        basically indexing them
    F test for volatilities
    df = infinity
    compare critical value (df value) to computed f value
    validate results in excel'''
    pass


'''References for learning financial terms and concepts for metrics method: 
https://www.investopedia.com/terms/v/volatility.asp
https://www.investopedia.com/articles/04/092904.asp
https://www.investopedia.com/terms/s/sharperatio.asp
https://www.investopedia.com/terms/d/downside-deviation.asp
https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
'''

'''VALIDATE RESULTS IN EXCEL (see if it's approx the same)'''


def metrics(koData, tbillData):
    '''Computes the annual volatility, 95% VaR, Sharpe Ratio, Downside Deviation
    and Maximum Drawdown for Coca-Cola (KO)'''
    # First, calculate the % daily returns of the stock
    dailyReturns = koData.pct_change() * 100
    stdDevDailyReturns = np.std(dailyReturns)
    # Compute historical volatility of the stock: standard deviation of daily returns * sqrt(number of periods on time horizon)
    koVolatility = stdDevDailyReturns * math.sqrt(len(koData))
    print('Coca-Cola\'s stock has a volatility of ', koVolatility,
          '%. This means that the stock\'s price can fluctuate up and down by', koVolatility, 'percent.')

    # Next, calculate the 95% VaR from historical simulation
    # First, sort the returns
    dailyReturnsSorted = dailyReturns.sort_values()
    # Then, calculate the VaR
    VaR95 = dailyReturnsSorted.quantile(0.05)
    print('Coca-Cola\'s stock has a 95% VaR of ', VaR95,
          '%. This means that we can say with 95% confidence that the stock\'s value has the potential to lose ', abs(VaR95), 'percent of its value over the course of the whole trading year.')

    # Next, calculate the annualized Sharpe Ratio for the stock: (mean daily return - mean risk free rate (treasury rate)) / stdDev(excess return) * sqrt(time period)
    # There is a NaN value at the first index of dailyReturns' values
    # because there is no prior date for the first index, need to remove the NaN value to calculate the mean
    meanDailyReturn = np.mean(
        dailyReturns.values[~(np.isnan(dailyReturns.values))])
    '''Risk free rate - based on Treasury Bill Rates
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2021
    Using 13 week coupon equivalent because it's short term and the 3 month short term treasury bill rate (TBIL), 
    a common risk free rate to use in calculations. Could use a shorter or longer duration (i.e. 4 weeks or 26 weeks), 
    but since 3 months is a commonly referenced risk free rate going with the 13 week.
    For Sharpe Ratio using the mean of the risk free rate'''
    meanRiskFreeRate = np.mean(tbillData.values)
    dailySharpeRatio = (meanDailyReturn - meanRiskFreeRate) / \
        stdDevDailyReturns
    # The above formula only calculates the daily Sharpe Ratio, need to convert it to the annual by multiplying by sqrt(# of trading days)
    annualizedSharpeRatio = dailySharpeRatio * math.sqrt(len(koData))
    print('Coca-Cola\'s stock had an annual Sharpe Ratio of ', annualizedSharpeRatio,
          '. This means that the stock will have significant risk of loss.')

    # Next, calculate the Downside Deviation
    # Will utilize the tbill rate as the minimum acceptable return
    downsideDevDailyReturns = dailyReturns.values[~(
        np.isnan(dailyReturns.values))]
    downsideDifferences = downsideDevDailyReturns - tbillData.values
    negativeDownsides = downsideDifferences[downsideDifferences < 0]
    downsideDeviation = math.sqrt(
        sum(np.square(negativeDownsides))/len(downsideDevDailyReturns))
    print('Coca-Cola\'s stock had a Downside Deviation of ', downsideDeviation,
          '. This means that the stock will have little risk of loss.')

    # Finally, calculate the Maximum Drawdown


'''References for learning financial terms and concepts for capm method:
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
https://www.investopedia.com/terms/a/alpha.asp
'''


def capm(tslaData, spyData):
    '''Calculates the Beta and Alpha for Tesla (TSLA)'''
    # First, calculate Beta = covariance/ market variance
    tslaDailyReturns = tslaData.pct_change() * 100
    spyDailyReturns = spyData.pct_change() * 100
    tslaSeries = pd.Series(tslaDailyReturns)
    spySeries = pd.Series(spyDailyReturns)
    covariance = tslaSeries.cov(spySeries)
    marketVariance = spySeries.var()
    beta = covariance/marketVariance
    print('Tesla had a beta of ', beta, '. This means that it is ',
          (beta-1)*100, '% more volatile than the S&P 500.')
    # Next calculate the alpha = stock annual return - benchmark annual return
    print(tslaData.values[0], tslaData.values[-1])
    tslaAnnualReturn = (
        tslaData.values[-1]-tslaData.values[0])/tslaData.values[0]
    spyAnnualReturn = (spyData.values[-1]-spyData.values[0])/spyData.values[0]
    alpha = tslaAnnualReturn - spyAnnualReturn
    print('Tesla had an alpha of ', alpha, '. This means that it has a ',
          alpha, '% higher return rate than the S&P 500.')


def main():
    '''Method to run the different methods'''
    # Generate the DataFrame with adjusted closing price history data
    closeHistoryDF = generateDataframe()
    # Get Treasury Bill Rates (TBIL) for calculating the Risk-free rate in the metrics method
    # Data from https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2021
    tbillDF = pd.read_csv('daily-treasury-rates.csv')
    tbillDF = tbillDF[['13 WEEKS COUPON EQUIVALENT']]
    print('Finished generating DataFrames')
    # print(closeHistoryDF.head()) # See the head of the DataFrame
    # Checked for any NaN values, all had 0
    # print(closeHistoryDF['KO Adj Close'].isna().sum())
    # print(closeHistoryDF['TSLA Adj Close'].isna().sum())
    # print(closeHistoryDF['SPY Adj Close'].isna().sum())

    # Execute the 3 methods
    '''NEED TO GRAPH DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    print('STATISTICS METHOD OUTPUT:')
    statistics(closeHistoryDF.loc[:, "KO Adj Close"],
               closeHistoryDF.loc[:, "TSLA Adj Close"])
    print('METRICS METHOD OUTPUT')
    metrics(closeHistoryDF.loc[:, "KO Adj Close"],
            tbillDF.loc[:, '13 WEEKS COUPON EQUIVALENT'])
    print('CAPM METHOD OUTPUT:')
    capm(closeHistoryDF.loc[:, "TSLA Adj Close"],
         closeHistoryDF.loc[:, "SPY Adj Close"])


if __name__ == '__main__':
    main()
