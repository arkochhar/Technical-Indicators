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

def HA(df, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Heiken Ashi Candles (HA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Heiken Ashi Close (HA_$ohlc[3])
            Heiken Ashi Open (HA_$ohlc[0])
            Heiken Ashi High (HA_$ohlc[1])
            Heiken Ashi Low (HA_$ohlc[2])
    """

    ha_open = 'HA_' + ohlc[0]
    ha_high = 'HA_' + ohlc[1]
    ha_low = 'HA_' + ohlc[2]
    ha_close = 'HA_' + ohlc[3]
    
    df[ha_close] = (df[ohlc[0]] + df[ohlc[1]] + df[ohlc[2]] + df[ohlc[3]]) / 4

    df[ha_open] = 0.00
    for i in range(0, len(df)):
        if i == 0:
            df[ha_open].iat[i] = (df[ohlc[0]].iat[i] + df[ohlc[3]].iat[i]) / 2
        else:
            df[ha_open].iat[i] = (df[ha_open].iat[i - 1] + df[ha_close].iat[i - 1]) / 2
            
    df[ha_high]=df[[ha_open, ha_close, ohlc[1]]].max(axis=1)
    df[ha_low]=df[[ha_open, ha_close, ohlc[2]]].min(axis=1)

    return df

def SMA(df, base, target, period):
    """
    Function to compute Simple Moving Average (SMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the SMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    df[target] = df[base].rolling(window=period).mean()
    df[target].fillna(0, inplace=True)

    return df

def STDDEV(df, base, target, period):
    """
    Function to compute Standard Deviation (STDDEV)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the SMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    df[target] = df[base].rolling(window=period).std()
    df[target].fillna(0, inplace=True)

    return df

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
    if not 'TR' in df.columns:
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

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
       
    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] <  df['final_lb'].iat[i] else 0.00 
                 
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    
    df.fillna(0, inplace=True)

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

def BBand(df, base='Close', period=20, multiplier=2):
    """
    Function to compute Bollinger Band (BBand)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the SD
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Upper Band (UpperBB_$period_$multiplier)
            Lower Band (LowerBB_$period_$multiplier)
    """
    
    upper = 'UpperBB_' + str(period) + '_' + str(multiplier)
    lower = 'LowerBB_' + str(period) + '_' + str(multiplier)
    
    sma = df[base].rolling(window=period, min_periods=period - 1).mean()
    sd = df[base].rolling(window=period).std()
    df[upper] = sma + (multiplier * sd)
    df[lower] = sma - (multiplier * sd)
    
    df[upper].fillna(0, inplace=True)
    df[lower].fillna(0, inplace=True)
    
    return df

def RSI(df, base="Close", period=21):
    """
    Function to compute Relative Strength Index (RSI)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Relative Strength Index (RSI_$period)
    """
 
    delta = df[base].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    rUp = up.ewm(com=period - 1,  adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

    df['RSI_' + str(period)] = 100 - 100 / (1 + rUp / rDown)
    df['RSI_' + str(period)].fillna(0, inplace=True)

    return df

def Ichimoku(df, ohlc=['Open', 'High', 'Low', 'Close'], param=[9, 26, 52, 26]):
    """
    Function to compute Ichimoku Cloud parameter (Ichimoku)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        param: Periods to be used in computation (default [tenkan_sen_period, kijun_sen_period, senkou_span_period, chikou_span_period] = [9, 26, 52, 26])
        
    Returns :
        df : Pandas DataFrame with new columns added for ['Tenkan Sen', 'Kijun Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']
    """
    
    high = df[ohlc[1]]
    low = df[ohlc[2]]
    close = df[ohlc[3]]
    
    tenkan_sen_period = param[0]
    kijun_sen_period = param[1]
    senkou_span_period = param[2]
    chikou_span_period = param[3]
    
    tenkan_sen_column = 'Tenkan Sen'
    kijun_sen_column = 'Kijun Sen'
    senkou_span_a_column = 'Senkou Span A'
    senkou_span_b_column = 'Senkou Span B'
    chikou_span_column = 'Chikou Span'
    
    # Tenkan-sen (Conversion Line)
    tenkan_sen_high = high.rolling(window=tenkan_sen_period).max()
    tenkan_sen_low = low.rolling(window=tenkan_sen_period).min()
    df[tenkan_sen_column] = (tenkan_sen_high + tenkan_sen_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen_high = high.rolling(window=kijun_sen_period).max()
    kijun_sen_low = low.rolling(window=kijun_sen_period).min()
    df[kijun_sen_column] = (kijun_sen_high + kijun_sen_low) / 2
    
    # Senkou Span A (Leading Span A)
    df[senkou_span_a_column] = ((df[tenkan_sen_column] + df[kijun_sen_column]) / 2).shift(kijun_sen_period)
    
    # Senkou Span B (Leading Span B)
    senkou_span_high = high.rolling(window=senkou_span_period).max()
    senkou_span_low = low.rolling(window=senkou_span_period).min()
    df[senkou_span_b_column] = ((senkou_span_high + senkou_span_low) / 2).shift(kijun_sen_period)
    
    # The most current closing price plotted chikou_span_period time periods behind
    df[chikou_span_column] = close.shift(-1 * chikou_span_period)
    
    return df