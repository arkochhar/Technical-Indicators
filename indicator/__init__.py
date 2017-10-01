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

def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])
    
    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()
    
    df[target].fillna(0, inplace=True)
    return df

def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Average True Range (ATR)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'tr' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())
        
        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
        
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)
    
    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=True)
    
    return df

def SuperTrend(df, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute SuperTrend
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)
    
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
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]
   
    index = df.index.name
    df.reset_index(inplace=True)    
    
    # Compute final upper and lower bands
    for i in range(0, len(df)):
        if i < period:
            df.set_value(i, 'basic_ub', 0.00)
            df.set_value(i, 'basic_lb', 0.00)
            df.set_value(i, 'final_ub', 0.00)
            df.set_value(i, 'final_lb', 0.00)
        else:
            df.set_value(i, 'final_ub', (df.get_value(i, 'basic_ub') 
                                         if df.get_value(i, 'basic_ub') < df.get_value(i-1, 'final_ub') or df.get_value(i-1, ohlc[3]) > df.get_value(i-1, 'final_ub') 
                                         else df.get_value(i-1, 'final_ub')))
            df.set_value(i, 'final_lb', (df.get_value(i, 'basic_lb') 
                                         if df.get_value(i, 'basic_lb') > df.get_value(i-1, 'final_lb') or df.get_value(i-1, ohlc[3]) < df.get_value(i-1, 'final_lb') 
                                         else df.get_value(i-1, 'final_lb')))
    
    # Set the Supertrend value
    for i in range(0, len(df)):
        if i < period:
            df.set_value(i, st, 0.00)
        else:
            df.set_value(i, st, (df.get_value(i, 'final_ub')
                                 if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, ohlc[3]) <= df.get_value(i, 'final_ub')))
                                 else (df.get_value(i, 'final_lb')
                                       if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, ohlc[3]) > df.get_value(i, 'final_ub')))
                                       else (df.get_value(i, 'final_lb')
                                             if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, ohlc[3]) >= df.get_value(i, 'final_lb')))
                                             else (df.get_value(i, 'final_ub')
                                                   if((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, ohlc[3]) < df.get_value(i, 'final_lb')))
                                                   else 0.00
                                                  )
                                            )
                                      ) 
                                )
                        )

    df.set_index(index, inplace=True)
    
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    
    return df

def MACD(df, fastEMA=12, slowEMA=26, signal=9, base='Close'):
    """
    Function to compute Moving Average Convergence Divergence (MACD)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        fastEMA : Integer indicates faster EMA
        slowEMA : Integer indicates slower EMA
        signal : Integer indicates the signal generator for MACD
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        
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
    EMA(df, base, fE, fastEMA)
    EMA(df, base, sE, slowEMA)
    
    # Compute MACD
    df[macd] = np.where(np.logical_and(np.logical_not(df[fE] == 0), np.logical_not(df[sE] == 0)), df[fE] - df[sE], 0)
    
    # Compute MACD Signal
    EMA(df, macd, sig, signal)
    
    # Compute MACD Histogram
    df[hist] = np.where(np.logical_and(np.logical_not(df[macd] == 0), np.logical_not(df[sig] == 0)), df[macd] - df[sig], 0)
    
    return df

# Unit Test
if __name__ == '__main__':
    import quandl
    import time
    
    df = quandl.get("NSE/NIFTY_50")
 
    # unit test EMA algorithm
    def test_ema(colFrom = 'Close', test_period = 5, ignore = 0, forATR=False):
        colTo = 'ema_' + str(test_period)
        
        # Actual function to test compare
        print('EMA Test with alpha {}'.format(forATR))
        start = time.time()
        EMA(df, colFrom, colTo, test_period, alpha=forATR)
        end = time.time()
        print('Time take by Pandas computations for EMA {}'.format(end-start))

        start = time.time()
        coef = 2 / (test_period + 1)
        periodTotal = 0
        colTest = colTo + '_test'
        df.reset_index(inplace=True)
        for i in range(0, len(df)):
            if (i < ignore):
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                if (forATR == True):
                    df.set_value(i, colTest, (((df.get_value(i - 1, colTest) * (test_period - 1)) + df.get_value(i, colFrom)) / test_period))
                else:
                    df.set_value(i, colTest, (((df.get_value(i, colFrom) - df.get_value(i - 1, colTest)) * coef) + df.get_value(i - 1, colTest)))
        
        df.set_index('Date', inplace=True)
        end = time.time()
        print('Time take by manual computations for EMA {}'.format(end-start))
 
        df[colTest + '_check'] = df[colTo].round(6) == df[colTest].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[colTest + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[colTest + '_check'].sum() / len(df)) * 100, 2)))
    
    def test_ATR():
        print('ATR Test')
        start = time.time()
        ATR(df, 7)
        end = time.time()
        print('Time take by Pandas computations for ATR {}'.format(end-start))

        start = time.time()
        ignore=0
        test_period = 7
        start = time.time()
        periodTotal = 0
        colFrom = 'TR'
        colTo = 'ATR_7'
        colTest = colTo + '_test'

        df['h-l'] = df['High'] - df['Low']
        df['h-yc'] = abs(df['High'] - df['Close'].shift())
        df['l-yc'] = abs(df['Low'] - df['Close'].shift())
        
        df[colFrom] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1) 
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

        df.reset_index(inplace=True)
        for i in range(0, len(df)):
            if (i < ignore):
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                df.set_value(i, colTest, (((df.get_value(i - 1, colTest) * (test_period - 1)) + df.get_value(i, colFrom)) / test_period))    
        df.set_index('Date', inplace=True)
        end = time.time()
        print('Time take by manual computations for ATR {}'.format(end-start))
        
        df[colTest + '_check'] = df[colTo].round(6) == df[colTest].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[colTest + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[colTest + '_check'].sum() / len(df)) * 100, 2)))

    def test_MACD():
        print('MACD Test')
        start = time.time()
        MACD(df)
        end = time.time()
        print('Time take by Pandas computations for MACD {}'.format(end-start))

        df.reset_index(inplace=True)

        fastEMA = 12
        slowEMA = 26
        signal = 9
        fE = "ema_" + str(fastEMA)
        sE = "ema_" + str(slowEMA)
        macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

        fE_test = fE + '_test'
        sE_test = sE + '_test'
        macd_test = macd + '_test'
        sig_test = sig + '_test'
        hist_test = hist + '_test'
    
        colFrom = 'Close'

        start = time.time()
        # Compute fast EMA    
        #EMA(df, base, fE, fastEMA)
        periodTotal = 0
        ignore = 0
        test_period = fastEMA
        colTest = fE_test
        coef = 2 / (test_period + 1)
        for i in range(0, len(df)):
            if (i < ignore):
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                df.set_value(i, colTest, (((df.get_value(i, colFrom) - df.get_value(i - 1, colTest)) * coef) + df.get_value(i - 1, colTest)))
        
        # Compute slow EMA    
        #EMA(df, base, sE, slowEMA)
        periodTotal = 0
        ignore = 0
        test_period = slowEMA
        colTest = sE_test
        coef = 2 / (test_period + 1)
        for i in range(0, len(df)):
            if (i < ignore):
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                df.set_value(i, colTest, (((df.get_value(i, colFrom) - df.get_value(i - 1, colTest)) * coef) + df.get_value(i - 1, colTest)))
        
        # Compute MACD
        df[macd_test] = np.where(np.logical_and(np.logical_not(df[fE_test] == 0), np.logical_not(df[sE_test] == 0)), df[fE_test] - df[sE_test], 0)
        
        # Compute MACD Signal
        #EMA(df, macd, sig, signal, slowEMA - 1)
        periodTotal = 0
        ignore = 0
        test_period = signal
        colTest = sig_test
        colFrom = macd_test
        coef = 2 / (test_period + 1)
        for i in range(0, len(df)):
            if (i < ignore):
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period - 1):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, 0.00)
            elif (i < ignore + test_period):
                periodTotal += df.get_value(i, colFrom)
                df.set_value(i, colTest, (periodTotal / test_period))
            else:
                df.set_value(i, colTest, (((df.get_value(i, colFrom) - df.get_value(i - 1, colTest)) * coef) + df.get_value(i - 1, colTest)))
        
        # Compute MACD Histogram
        df[hist_test] = np.where(np.logical_and(np.logical_not(df[macd_test] == 0), np.logical_not(df[sig_test] == 0)), df[macd_test] - df[sig_test], 0)

        end = time.time()
        print('Time take by manual computations for MACD {}'.format(end-start))

        df.set_index('Date', inplace=True)

        print('MACD Stats')
        df[macd + '_check'] = df[macd].round(6) == df[macd_test].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[macd + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[macd + '_check'].sum() / len(df)) * 100, 2)))
        print('Signal Stats')
        df[sig + '_check'] = df[sig].round(6) == df[sig_test].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[sig + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[sig + '_check'].sum() / len(df)) * 100, 2)))
        print('Hist Stats')
        df[hist + '_check'] = df[hist].round(6) == df[hist_test].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[hist + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[hist + '_check'].sum() / len(df)) * 100, 2)))
        
    def test_SuperTrend():
        period = 7
        multiplier = 2
        atr = 'ATR_' + str(period)
        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)
        st_test = st + '_test'
        stx_test = stx + '_test'

        print('SuperTrend Test')
        start = time.time()
        SuperTrend(df, period, multiplier)
        end = time.time()
        print('Time take by Pandas computations for SuperTrend {}'.format(end-start))
        
        start = time.time()
        ATR(df, period)
        df['basic_ub'] = (df['High'] + df['Low']) / 2 + multiplier * df[atr]
        df['basic_lb'] = (df['High'] + df['Low']) / 2 - multiplier * df[atr]
       
        index = df.index.name
        df.reset_index(inplace=True)
        
        # Compute final upper and lower bands
        for i in range(0, len(df)):
            if i < period:
                df.set_value(i, 'basic_ub', 0.00)
                df.set_value(i, 'basic_lb', 0.00)
                df.set_value(i, 'final_ub', 0.00)
                df.set_value(i, 'final_lb', 0.00)
            else:
                df.set_value(i, 'final_ub', (df.get_value(i, 'basic_ub') 
                                             if df.get_value(i, 'basic_ub') < df.get_value(i-1, 'final_ub') or df.get_value(i-1, 'Close') > df.get_value(i-1, 'final_ub') 
                                             else df.get_value(i-1, 'final_ub')))
                df.set_value(i, 'final_lb', (df.get_value(i, 'basic_lb') 
                                             if df.get_value(i, 'basic_lb') > df.get_value(i-1, 'final_lb') or df.get_value(i-1, 'Close') < df.get_value(i-1, 'final_lb') 
                                             else df.get_value(i-1, 'final_lb')))
        
        # Set the Supertrend value
        for i in range(0, len(df)):
            if i < period:
                df.set_value(i, st_test, 0.00)
            else:
                df.set_value(i, st_test, (df.get_value(i, 'final_ub')
                                     if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'Close') <= df.get_value(i, 'final_ub')))
                                     else (df.get_value(i, 'final_lb')
                                           if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'Close') > df.get_value(i, 'final_ub')))
                                           else (df.get_value(i, 'final_lb')
                                                 if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'Close') >= df.get_value(i, 'final_lb')))
                                                 else (df.get_value(i, 'final_ub')
                                                       if((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'Close') < df.get_value(i, 'final_lb')))
                                                       else 0.00
                                                      )
                                                )
                                          ) 
                                    )
                            )
    
        df.set_index(index, inplace=True)
        
        # Mark the trend direction up/down
        df[stx_test] = np.where((df[st_test] > 0.00), np.where((df['Close'] < df[st_test]), 'down',  'up'), np.NaN)
    
        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
        end = time.time()
        print('Time take by manual computations for SuperTrend {}'.format(end-start))

        df[st + '_check'] = df[st].round(6) == df[st_test].round(6)
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[st + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[st + '_check'].sum() / len(df)) * 100, 2)))
        print('Hist Stats')
        df[stx + '_check'] = df[stx] == df[stx_test]
        print('\tTotal Rows: {}'.format(len(df)))
        print('\tColumns Match: {}'.format(df[stx + '_check'].sum()))
        print('\tSuccess Rate: {}%'.format(round((df[stx + '_check'].sum() / len(df)) * 100, 2)))

    #test_ema()
    #test_ema(forATR=True)
    #test_ATR()
    #test_MACD()
    test_SuperTrend()
    
    #print(df.head(10))
    #print(df.tail(10))