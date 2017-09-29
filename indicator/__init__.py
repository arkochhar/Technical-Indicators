#!python
"""
Project: Technical Indicators
Package: indicator
Author: Anuraag Rai Kochhar
Email: arkochhar@hotmail.com
Repository: https://github.com/arkochhar/Technical-Indicators
Version: 1.0.0
License: GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007
"""

# External dependencies
import numpy as np
import pandas as pd

"""
Usage : 
    data = {
        "data": {
            "candles": [
                ["05-09-2013", 5553.75, 5625.75, 5552.700195, 5592.950195, 274900],
                ["06-09-2013", 5617.450195, 5688.600098, 5566.149902, 5680.399902, 253000],
                ["10-09-2013", 5738.5, 5904.850098, 5738.200195, 5896.75, 275200],
                ["11-09-2013", 5887.25, 5924.350098, 5832.700195, 5913.149902, 265000],
                ["12-09-2013", 5931.149902, 5932, 5815.799805, 5850.700195, 273000],
                ...
                ["27-01-2014", 6186.299805, 6188.549805, 6130.25, 6135.850098, 190400],
                ["28-01-2014", 6131.850098, 6163.600098, 6085.950195, 6126.25, 184100],
                ["29-01-2014", 6161, 6170.450195, 6109.799805, 6120.25, 146700],
                ["30-01-2014", 6067, 6082.850098, 6027.25, 6073.700195, 208100],
                ["31-01-2014", 6082.75, 6097.850098, 6067.350098, 6089.5, 146700]        
            ]
        }
    }
    
    # Date must be present as a Pandas DataFrame with ['date', 'open', 'high', 'low', 'close', 'volume'] as columns
    df = pd.DataFrame(data["data"]["candles"], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Columns as added by each function specific to their computations
    EMA(df, 'close', 'ema_5', 5)
    ATR(df, 14)
    SuperTrend(df, 10, 3)
    MACD(df)
"""

def EMA(df, columnFrom, columnTo, period, linesToIgnore=0, forATR=False):
    """
    Function to compute Exponential Moving Average (EMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        columnFrom : String indicating the column name from which the EMA needs to be computed from
        columnTo : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        linesToIgnore : Integer indicates the rows to ignore before the row from where the computation needs to begin (default is 0)
        forATR : Boolean if True indicates to use the formula for computing EMA for ATR purposes (default is False)
        
    Returns :
        df : Pandas DataFrame with new column added with name columnTo
    """
    
    if (forATR == True):
        df[columnTo] = df[columnFrom].ewm(com=period, min_periods=period+linesToIgnore).mean()
    else:
        df[columnTo] = df[columnFrom].ewm(span=period, min_periods=period+linesToIgnore).mean()
    
    return df

def ATR(df, period):
    """
    Function to compute Average True Range (ATR)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (tr)
            ATR (atr_$period)
    """
    atr = 'atr_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'tr' in df.columns:
        df['h-l'] = df['high'] - df['low']
        df['h-yc'] = abs(df['high'] - df['close'].shift())
        df['l-yc'] = abs(df['low'] - df['close'].shift())
        
        df['tr'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
        
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)
    
    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'tr', atr, period, linesToIgnore=1, forATR=True)
    
    return df

def SuperTrend(df, period, multiplier):
    """
    Function to compute SuperTrend
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (tr), ATR (atr_$period)
            SuperTrend (st_$period_$multiplier)
            SuperTrend Direction (stx_$period_$multiplier)
    """

    ATR(df, period)
    atr = 'atr_' + str(period)
    st = 'st_' + str(period) + '_' + str(multiplier)
    stx = 'stx_' + str(period) + '_' + str(multiplier)
    
    """
    SuperTrend Algorithm :
    
        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
        
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """
    
    # Compute basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df[atr]
   
    # Compute final upper and lower bands
    for i in range(0, len(df)):
        if i < period:
            df.set_value(i, 'basic_ub', 0.00)
            df.set_value(i, 'basic_lb', 0.00)
            df.set_value(i, 'final_ub', 0.00)
            df.set_value(i, 'final_lb', 0.00)
        else:
            df.set_value(i, 'final_ub', (df.get_value(i, 'basic_ub') 
                                         if df.get_value(i, 'basic_ub') < df.get_value(i-1, 'final_ub') or df.get_value(i-1, 'close') > df.get_value(i-1, 'final_ub') 
                                         else df.get_value(i-1, 'final_ub')))
            df.set_value(i, 'final_lb', (df.get_value(i, 'basic_lb') 
                                         if df.get_value(i, 'basic_lb') > df.get_value(i-1, 'final_lb') or df.get_value(i-1, 'close') < df.get_value(i-1, 'final_lb') 
                                         else df.get_value(i-1, 'final_lb')))
    
    # Set the Supertrend value
    for i in range(0, len(df)):
        if i < period:
            df.set_value(i, st, 0.00)
        else:
            df.set_value(i, st, (df.get_value(i, 'final_ub')
                                 if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'close') <= df.get_value(i, 'final_ub')))
                                 else (df.get_value(i, 'final_lb')
                                       if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'close') > df.get_value(i, 'final_ub')))
                                       else (df.get_value(i, 'final_lb')
                                             if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'close') >= df.get_value(i, 'final_lb')))
                                             else (df.get_value(i, 'final_ub')
                                                   if((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'close') < df.get_value(i, 'final_lb')))
                                                   else 0.00
                                                  )
                                            )
                                      ) 
                                )
                        )
    
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    
    return df

def MACD(df, fastEMA=12, slowEMA=26, signal=9):
    """
    Function to compute Moving Average Convergence Divergence (MACD)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        fastEMA : Integer indicates faster EMA
        slowEMA : Integer indicates slower EMA
        signal : Integer indicates the signal generator for MACD
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Fast EMA (ema_$fastEMA)
            Slow EMA (ema_$slowEMA)
            MACD (macd_$fastEMA_$slowEMA_$signal)
            MACD Signal (signal_$fastEMA_$slowEMA_$signal)
            MACD Histogram (MACD (hist_$fastEMA_$slowEMA_$signal)) 
    """

    fE = "ema_" + str(fastEMA)
    sE = "ema_" + str(slowEMA)
    macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

    # Compute fast and slow EMA    
    EMA(df, 'close', fE, fastEMA)
    EMA(df, 'close', sE, slowEMA)
    
    # Compute MACD
    df[macd] = np.where(np.logical_and(np.logical_not(df[fE] == 0), np.logical_not(df[sE] == 0)), df[fE] - df[sE], 0)
    
    # Compute MACD Signal
    EMA(df, macd, sig, signal, slowEMA - 1)
    
    # Compute MACD Histogram
    df[hist] = np.where(np.logical_and(np.logical_not(df[macd] == 0), np.logical_not(df[sig] == 0)), df[macd] - df[sig], 0)
    
    return df

# Unit Test
if __name__ == '__main__':
    import quandl
    df = quandl.get("NSE/NIFTY_50")
    df.reset_index(inplace=True)
 
    # unit test EMA algorithm
    def test_ema(forATR=False):
        colFrom = 'Close'
        test_period = 5
        linesToIgnore = 0
        colTo = 'ema_' + str(test_period)
    
        # Actual function to test compare
        EMA(df, colFrom, colTo, test_period, linesToIgnore=linesToIgnore, forATR=forATR)
    
        coef = 2 / (test_period + 1)
        periodTotal = 0
        colTest = colTo + '_test'
        for i in range(0, len(df)):
            if (i < linesToIgnore):
                df.set_value(i, colTest, 0.00)
            elif (i < linesToIgnore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < linesToIgnore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                if (forATR == True):
                    df.set_value(i, colTest, (((df.get_value(i - 1, colTest) * (test_period - 1)) + df.get_value(i, colFrom)) / test_period))
                else:
                    df.set_value(i, colTest, (((df.get_value(i, colFrom) - df.get_value(i - 1, colTest)) * coef) + df.get_value(i - 1, colTest)))
         
        df[colTest + '_check'] = df[colTo].round(6) == df[colTest].round(6)
        print('EMA Test with ATR {}'.format(forATR))
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[colTest + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[colTest + '_check'].sum() / len(df)) * 100, 2)))
    
    test_ema()
    #print(df.head(10))
    #print(df.tail(10))
    test_ema(True)
    #print(df.head(10))
    #print(df.tail(10))
    
    #ATR(df, 14)
    #SuperTrend(df, 10, 3)
    #MACD(df)