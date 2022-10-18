# Zakariyya Scavotto Fall 2022 SSMIF Coding Challenge

# Import listed libraries from instructions for data calculations and visualization
from webbrowser import get
import numpy as np
import pandas as pd
from scipy.stats import skewtest, f, ttest_ind
import matplotlib.pyplot as plt
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


'''References for learning statistical tests:
https://www.statology.org/welchs-t-test/
https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/f-test/'''


def statistics(koData, tslaData):
    '''Computes statistical test on the two equities (KO and TSLA) to see if there is a statistically
    significant difference in the mean daily returns or the volatilities'''
    koDailyReturns, tslaDailyReturns = (
        koData.pct_change() * 100).dropna(), (tslaData.pct_change() * 100).dropna()
    # Uncomment the plt.show() calls to see the graphs, both distributions appear to be normally distributed
    plt.hist(koDailyReturns, bins='rice', label='Coke Daily Close')
    plt.legend()
    # plt.show()
    plt.hist(tslaDailyReturns, bins='rice', label='Tesla Daily Close')
    plt.legend()
    # plt.show()
    plt.boxplot(koDailyReturns, labels=["Coke Daily close price"])
    # plt.show
    plt.boxplot(tslaDailyReturns, labels=["Tesla Daily close price"])
    # plt.show()

    '''First test - test for statistically sig. difference between mean daily returns
    I have decided to use Welch's t-test since it tests if two populations have equal means, and is more reliable when samples may have unequal variances
    which was apparent looking at the histograms of the distributions (and will be checked in the test comparing their volatilities)'''

    print('To test for a significant difference in mean daily returns, I used a modification of a student\'s t-test Welch\'s t-test. To perform this test, I am assuming that the observations in each sample are:')
    print('Normally distributed (confirmed looking at histograms). Note: for a student\'s t-test, there is an assumption of equal variance between the two grooups, but this assumption is not the case in Welch\'s t-test, meaning that the only assumption is the normal distribution assumption.\n')
    # compute Welch's t-test on the two distributions of daily returns, and save the p-value
    tTestP = ttest_ind(koDailyReturns.values,
                       tslaDailyReturns.values, equal_var=False).pvalue
    print('p-value: ', tTestP, '\n')
    if tTestP < 0.05:
        print('result: because the p-value is less than the significance level of 0.05, we can reject the null hypothesis, meaning that there is a statistically significant difference between the mean daily returns of the two equities.\n')
    else:
        print('result: because the p-value is greater than the significance level of 0.05, we fail to reject the null hypothesis, meaning that there is no statistically significant difference between the mean daily returns of the two equities.\n')

    '''Second test - test for statistically significant difference in volatilities
    I have decided to use a f-test, since it compares the variances between the two different data sets'''

    print('To test for a significant difference in volatilities, I used a f-test. To perform this test, I am assuming that the observations in each sample are:')
    print('Normally distributed (confirmed looking at histograms) and are independent events (Coke is a beverage company, while Tesla is a car company, one does not have a direct impact on the other).\n')
    fStatistic = np.var(tslaDailyReturns.values, ddof=1) / \
        np.var(koDailyReturns.values, ddof=1)  # Calculate f-statistic
    dfn = koDailyReturns.values.size-1  # Degrees of freedom in numerator
    dfd = tslaDailyReturns.values.size-1  # Degrees of freedom in denominator
    fTestP = 1-f.cdf(fStatistic, dfn, dfd)  # Find p-value of F test statistic
    print('p-value: ', fTestP, '\n')
    if fTestP < 0.05:
        print('result: because the p-value is less than the significance level of 0.05, we can reject the null hypothesis, meaning that there is a statistically significant difference between the volatilities of the two equities.')
    else:
        print('result: because the p-value is greater than the significance level of 0.05, we fail to reject the null hypothesis, meaning that there is no statistically significant difference between the volatilities of the two equities.')


'''References for learning financial terms and concepts for metrics method: 
https://www.investopedia.com/terms/v/volatility.asp
https://www.investopedia.com/articles/04/092904.asp
https://www.investopedia.com/terms/s/sharperatio.asp
https://www.investopedia.com/terms/d/downside-deviation.asp
https://www.investopedia.com/terms/m/maximum-drawdown-mdd.as'''


def getPerecentDailyReturns(data):
    '''Gets the percent daily returns of the stock (how much the price moved up/down relative to the previous day)'''
    return (data.pct_change() * 100).dropna()


def getAnnualizedVolatility(data):
    '''Calculates the historical volatility of the given stock over the course of the whole year'''
    # First, calculate the % daily returns of the stock and the standard deviation
    dailyReturns = getPerecentDailyReturns(data)
    stdDevReturns = np.std(dailyReturns)
    # Compute historical volatility: standard deviation of daily returns * sqrt(number of periods on time horizon)
    return stdDevReturns * math.sqrt(len(data))  # using 252 since


def get95PcntVaR(data):
    '''Computes the 95% Value at Risk (VaR) from historical simulation of the given stock'''
    # First, get the % daily returns of the stock and sort them
    dailyReturns = getPerecentDailyReturns(data)
    dailyReturnsSorted = dailyReturns.sort_values()
    # Then, calculate the VaR with a 95% confidence level by finding the value that returns the lowest 5% of historical returns
    return dailyReturnsSorted.quantile(0.05)


def getAnnualizedSharpeRatio(data, tbillData):
    '''Computes the annualized Sharpe Ratio for the given stock'''
    # Get the daily returns, the mean daily return, as well as the standard deviation of the returns
    dailyReturns = getPerecentDailyReturns(data)
    meanDailyReturn = np.mean(dailyReturns)
    stdDevDailyReturns = np.std(dailyReturns)
    meanRiskFreeRate = np.mean(tbillData)
    # Calculate daily Sharpe Ratio: : (mean daily return - mean risk free rate (aka the treasury rate)) / stdDev(excess return)
    dailySharpeRatio = (meanDailyReturn - meanRiskFreeRate) / \
        stdDevDailyReturns
    # The above formula only calculates the daily Sharpe Ratio, need to convert it to the annual by multiplying by sqrt(# of trading days)
    return dailySharpeRatio * math.sqrt(len(data))


def getDownsideDeviation(data, tbillData):
    '''Calculates the Downside Deviation of the given stock, utilizing the tbill rate as the minimum acceptable return'''
    dailyReturns = getPerecentDailyReturns(data)
    # Gets the downside differences by comparing the daily returns the minimum acceptable return (tbill rate)
    downsideDifferences = dailyReturns - tbillData.values
    # isolates the negative downsides
    negativeDownsides = downsideDifferences[downsideDifferences < 0]
    # return the downside deviation: sqrt(sum of square of the negative downsides / number of periods)
    return math.sqrt(sum(np.square(negativeDownsides))/len(data))


def getMaximumDrawdown(data):
    '''Gets the maxmimum drawdown of the given stock as a %: ((lowestPrice-highestPrice)/highestPrice)*100'''
    highestPrice, lowestPrice = max(data.values), min(data.values)
    return ((lowestPrice-highestPrice)/highestPrice)*100


def metrics(koData, tbillData, spyData):
    '''Computes the annual volatility, 95% VaR, Sharpe Ratio, Downside Deviation and Maximum Drawdown for Coca-Cola (KO).
    I tried to make this method in a way where coke's data could be replaced by that of any other stock, to allow for versatility in possible future use'''
    # Compute the volatility of coke and S&P 500 and compare
    cokeVolatility, spyVolatility = getAnnualizedVolatility(
        koData), getAnnualizedVolatility(spyData)
    print('Coca-Cola\'s stock has a volatility of ', cokeVolatility,
          '%. This means that the stock\'s price can fluctuate up and down by', cokeVolatility, '%.')
    if cokeVolatility < spyVolatility:
        print('Coca-Cola\'s volatility is less than the volatility of the S&P 500, since its volatility is less than that of the S&P\'s ', spyVolatility, '%.\n')
    else:
        print('Coca-Cola\'s volatility is greater than the volatility of the S&P 500, since its volatility is greater than that of the S&P\'s ', spyVolatility, '%.\n')

    # Calculate the 95% VaR from historical simulation
    cokeVaR95, spyVaR95 = get95PcntVaR(koData), get95PcntVaR(spyData)
    print('Coca-Cola\'s stock has a 95% VaR of ', cokeVaR95,
          '%. This means that we can say with 95% confidence that the stock\'s value has the potential to lose ', abs(cokeVaR95), '% of its value over the course of the whole trading year.')
    if abs(cokeVaR95) < abs(spyVaR95):
        print('Coca-Cola\'s 95% VaR is less than the 95% VaR of the S&P 500, since its 95% VaR is less than that of the S&P\'s ',
              spyVaR95, '%, meaning that you would lose less investing in just coke than in the whole S&P 500.\n')
    else:
        print('Coca-Cola\'s 95% VaR is greater than the 95% VaR of the S&P 500, since its 95% VaR is greater than that of the S&P\'s ',
              spyVaR95, '%, meaning that you would lose more investing in just coke than in the whole S&P 500.\n')

    # Next, calculate the annualized Sharpe Ratio for the stock based on the Treasury Risk Free Rate (see below note)
    '''Risk free rate - based on Treasury Bill Rates
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2021
    Using 13 week coupon equivalent because it's short term and the 3 month short term treasury bill rate (TBIL), 
    a common risk free rate to use in calculations. Could use a shorter or longer duration (i.e. 4 weeks or 26 weeks), 
    but since 3 months is a commonly referenced risk free rate going with the 13 week.
    For Sharpe Ratio using the mean of the risk free rate'''
    cokeAnnualizedSharpeRatio, spyAnnualizedSharpeRatio = getAnnualizedSharpeRatio(
        koData, tbillData), getAnnualizedSharpeRatio(spyData, tbillData)
    print('Coca-Cola\'s stock had an annual Sharpe Ratio of ', cokeAnnualizedSharpeRatio,
          '. This means that the stock will have significant risk of loss.')
    if cokeAnnualizedSharpeRatio < spyAnnualizedSharpeRatio:
        print('Coca-Cola\'s 95% Sharpe Ratio is less than the Sharpe Ratio of the S&P 500, since its Sharpe Ratio is less than that of the S&P\'s ',
              spyAnnualizedSharpeRatio, '%, meaning that you would be better off investing in the whole S&P 500 than just coke.\n')
    else:
        print('Coca-Cola\'s 95% Sharpe Ratio is greater than the Sharpe Ratio of the S&P 500, since its Sharpe Ratio is greater than that of the S&P\'s ',
              spyAnnualizedSharpeRatio, '%, meaning that you would be better off investing in just coke than the whole S&P 500.\n')

    # Next, calculate the Downside Deviation
    # Will utilize the tbill rate as the minimum acceptable return
    cokeDownsideDeviation, spyDownsideDeviation = getDownsideDeviation(
        koData, tbillData), getDownsideDeviation(spyData, tbillData)
    print('Coca-Cola\'s stock had a Downside Deviation of ', cokeDownsideDeviation,
          '%. This means that the stock will have little risk of loss.')
    if cokeDownsideDeviation < spyDownsideDeviation:
        print('Coca-Cola\'s Downside Deviation is less than the Downside Deviation of the S&P 500, since its Downside Deviation is less than that of the S&P\'s ',
              spyDownsideDeviation, '%, meaning that you would be better off investing in coke than the whole S&P 500.\n')
    else:
        print('Coca-Cola\'s Downside Deviation is greater than the Downside Deviation of the S&P 500, since its Downside Deviation is greater than that of the S&P\'s ',
              spyDownsideDeviation, '%, meaning that you would be better off investing in the whole S&P 500 than just coke.\n')

    # Finally, calculate the Maximum Drawdown for KO based on the 2021 stock data
    cokeMaxDrawdown, spyMaxDrawdown = getMaximumDrawdown(
        koData), getMaximumDrawdown(spyData)
    print('Coca-Cola\'s stock had a Maximum Drawdown of ', cokeMaxDrawdown,
          '% This means that for someone or a company that invests in Coca-Cola, the maximum loss that they could get is a ', cokeMaxDrawdown, '% loss.')
    if abs(cokeMaxDrawdown) < abs(spyMaxDrawdown):
        print('Coca-Cola\'s Maximum Drawdown is less than the Maximum Drawdown of the S&P 500, since its Maximum Drawdown is less than that of the S&P\'s ',
              spyMaxDrawdown, '%, meaning that you would be better off investing in coke than the whole S&P 500.\n')
    else:
        print('Coca-Cola\'s Maximum Drawdown is greater than the Maximum Drawdown of the S&P 500, since its Maximum Drawdown is greater than that of the S&P\'s ',
              spyMaxDrawdown, '%, meaning that you would be better off investing in the whole S&P 500 than just coke.\n')


'''References for learning financial terms and concepts for capm method:
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
https://www.investopedia.com/terms/a/alpha.asp'''


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
          (beta-1)*100, '% more volatile than the S&P 500.\n')

    # Next calculate the alpha = stock annual return - benchmark annual return
    tslaAnnualReturn = (
        tslaData.values[-1]-tslaData.values[0])/tslaData.values[0]
    spyAnnualReturn = (spyData.values[-1]-spyData.values[0])/spyData.values[0]
    alpha = tslaAnnualReturn - spyAnnualReturn
    print('Tesla had an alpha of ', alpha, '. This means that it has a ',
          alpha, '% higher return rate than the S&P 500.\n')


def main():
    '''Method to run the different methods'''
    # Generate the DataFrame with adjusted closing price history data
    closeHistoryDF = generateDataframe()
    # Get Treasury Bill Rates (TBIL) for calculating the Risk-free rate in the metrics method and store in a DataFrame
    # Data from https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2021
    tbillDF = pd.read_csv('daily-treasury-rates.csv')
    tbillDF = tbillDF[['13 WEEKS COUPON EQUIVALENT']]
    print('Finished generating DataFrames, Equity and ETF DataFrame Head: ')
    print(closeHistoryDF.head())  # Uncomment to see the head of the DataFrame
    # Checked for any NaN values, all had 0
    # print(closeHistoryDF['KO Adj Close'].isna().sum())
    # print(closeHistoryDF['TSLA Adj Close'].isna().sum())
    # print(closeHistoryDF['SPY Adj Close'].isna().sum())

    # Execute the 3 methods
    print('\nSTATISTICS METHOD OUTPUT:')
    statistics(closeHistoryDF.loc[:, "KO Adj Close"],
               closeHistoryDF.loc[:, "TSLA Adj Close"])
    print()  # extra print for spacing
    print('METRICS METHOD OUTPUT')
    metrics(closeHistoryDF.loc[:, "KO Adj Close"],
            tbillDF.loc[:, '13 WEEKS COUPON EQUIVALENT'],
            closeHistoryDF.loc[:, "SPY Adj Close"])
    print()
    print('CAPM METHOD OUTPUT:')
    capm(closeHistoryDF.loc[:, "TSLA Adj Close"],
         closeHistoryDF.loc[:, "SPY Adj Close"])
    print('Program finished')


if __name__ == '__main__':
    main()
