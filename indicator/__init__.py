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
    
    periodTotal = 0
    coeff = 2 / (period + 1)
    
    for i in range(0, len(df)):
        if (i < linesToIgnore):
            # Ignores number of rows
            df.set_value(i, columnTo, 0.00)
        elif (i < linesToIgnore + period - 1):
            # Add rows to compute first Simple Moving Average
            periodTotal += df.get_value(i, columnFrom)
            df.set_value(i, columnTo, 0.00)
        elif (i < linesToIgnore + period):
            # Compute first Simple Moving Average
            periodTotal += df.get_value(i, columnFrom)
            df.set_value(i, columnTo, (periodTotal / period))
        else:
            # Compute EMA for each subsequent row
            if (forATR == True):
                # use formula for computing ATR
                df.set_value(i, columnTo, (0.1 * df.get_value(i, columnFrom) + (1 - 0.1) * df.get_value(i - 1, columnTo)))
            else:
                # use regular formula for computing EMA
                df.set_value(i, columnTo, (((df.get_value(i, columnFrom) - df.get_value(i - 1, columnTo)) * coeff) + df.get_value(i - 1, columnTo)))
        
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
"""
data = {
    "data": {
        "candles": [
            ["05-09-2013", 5553.75, 5625.75, 5552.700195, 5592.950195, 274900],
            ["06-09-2013", 5617.450195, 5688.600098, 5566.149902, 5680.399902, 253000],
            ["10-09-2013", 5738.5, 5904.850098, 5738.200195, 5896.75, 275200],
            ["11-09-2013", 5887.25, 5924.350098, 5832.700195, 5913.149902, 265000],
            ["12-09-2013", 5931.149902, 5932, 5815.799805, 5850.700195, 273000],
            ["13-09-2013", 5828, 5884.299805, 5822.899902, 5850.600098, 190800],
            ["16-09-2013", 5930.299805, 5957.25, 5798.149902, 5840.549805, 219500],
            ["17-09-2013", 5824.200195, 5857.799805, 5804.899902, 5850.200195, 157900],
            ["18-09-2013", 5872.75, 5916.899902, 5840.200195, 5899.450195, 153600],
            ["19-09-2013", 6044.149902, 6142.5, 6040.149902, 6115.549805, 287200],
            ["20-09-2013", 6104.549805, 6130.950195, 5932.850098, 6012.100098, 318600],
            ["23-09-2013", 5945.799805, 5989.399902, 5871.399902, 5889.75, 188300],
            ["24-09-2013", 5855, 5938.399902, 5854.549805, 5892.450195, 187600],
            ["25-09-2013", 5901.549805, 5910.549805, 5811.100098, 5873.850098, 200200],
            ["26-09-2013", 5872.799805, 5917.649902, 5864.100098, 5882.25, 220600],
            ["27-09-2013", 5905.549805, 5909.200195, 5819.299805, 5833.200195, 163100],
            ["30-09-2013", 5801.049805, 5810.200195, 5718.5, 5735.299805, 155700],
            ["01-10-2013", 5756.100098, 5786.450195, 5700.950195, 5780.049805, 159200],
            ["03-10-2013", 5819.100098, 5917.600098, 5802.700195, 5909.700195, 199800],
            ["04-10-2013", 5891.299805, 5950.450195, 5885, 5907.299805, 191500],
            ["07-10-2013", 5889.049805, 5912, 5825.850098, 5906.149902, 156400],
            ["08-10-2013", 5975, 5981.700195, 5913, 5928.399902, 158500],
            ["09-10-2013", 5893.25, 6015.5, 5877.100098, 6007.450195, 192000],
            ["10-10-2013", 6001.049805, 6033.950195, 5979.799805, 6020.950195, 159600],
            ["11-10-2013", 6104.850098, 6107.600098, 6046.399902, 6096.200195, 180800],
            ["14-10-2013", 6093, 6124.100098, 6082.899902, 6112.700195, 142200],
            ["15-10-2013", 6147.549805, 6156.299805, 6056.549805, 6089.049805, 218300],
            ["17-10-2013", 6098.5, 6110.75, 6032.549805, 6045.850098, 230600],
            ["18-10-2013", 6070.899902, 6201.450195, 6070.899902, 6189.350098, 250300],
            ["21-10-2013", 6202, 6218.950195, 6163.299805, 6204.950195, 197000],
            ["22-10-2013", 6192.299805, 6220.100098, 6181.799805, 6202.799805, 161900],
            ["23-10-2013", 6209.549805, 6217.950195, 6116.600098, 6178.350098, 188300],
            ["24-10-2013", 6162.799805, 6252.450195, 6142.950195, 6164.350098, 148900],
            ["25-10-2013", 6154, 6174.75, 6125.950195, 6144.899902, 144500],
            ["28-10-2013", 6155.100098, 6168.75, 6094.100098, 6101.100098, 146340700],
            ["29-10-2013", 6107.549805, 6228.049805, 6079.200195, 6220.899902, 197300],
            ["30-10-2013", 6230.799805, 6269.200195, 6222.600098, 6251.700195, 177600],
            ["31-10-2013", 6237.149902, 6309.049805, 6235.899902, 6299.149902, 239600],
            ["01-11-2013", 6289.75, 6332.600098, 6286.950195, 6307.200195, 191600],
            ["05-11-2013", 6282.149902, 6304.75, 6244.299805, 6253.149902, 181100],
            ["06-11-2013", 6260.549805, 6269.700195, 6208.700195, 6215.149902, 157100],
            ["07-11-2013", 6228.899902, 6288.950195, 6180.799805, 6187.25, 168800],
            ["08-11-2013", 6170.149902, 6185.149902, 6120.950195, 6140.75, 150100],
            ["11-11-2013", 6110.399902, 6141.649902, 6067.75, 6078.799805, 146100],
            ["12-11-2013", 6087.25, 6108.700195, 6011.75, 6018.049805, 153800],
            ["13-11-2013", 5998.850098, 6042.25, 5972.450195, 5989.600098, 162500],
            ["14-11-2013", 6037, 6101.649902, 6036.649902, 6056.149902, 154900],
            ["18-11-2013", 6111.049805, 6196.799805, 6110.399902, 6189, 164000],
            ["19-11-2013", 6197.25, 6212.399902, 6180.200195, 6203.350098, 161500],
            ["20-11-2013", 6186.850098, 6204.350098, 6106.950195, 6122.899902, 155800],
            ["21-11-2013", 6096.5, 6097.350098, 5985.399902, 5999.049805, 137200],
            ["22-11-2013", 6027.350098, 6049.600098, 5972.799805, 5995.450195, 134800],
            ["25-11-2013", 6035.950195, 6123.5, 6035.950195, 6115.350098, 127200],
            ["26-11-2013", 6099.25, 6112.700195, 6047.75, 6059.100098, 150700],
            ["27-11-2013", 6062.700195, 6074, 6030.299805, 6057.100098, 124900],
            ["28-11-2013", 6092, 6112.950195, 6068.299805, 6091.850098, 195300],
            ["29-11-2013", 6103.899902, 6182.5, 6103.799805, 6176.100098, 190700],
            ["02-12-2013", 6171.149902, 6228.700195, 6171.149902, 6217.850098, 145900],
            ["03-12-2013", 6204.25, 6225.399902, 6191.399902, 6201.850098, 156900],
            ["04-12-2013", 6187.950195, 6209.149902, 6149.899902, 6160.950195, 186200],
            ["05-12-2013", 6262.450195, 6300.549805, 6232, 6241.100098, 186100],
            ["06-12-2013", 6234.399902, 6275.350098, 6230.75, 6259.899902, 158500],
            ["09-12-2013", 6415, 6415.25, 6345, 6363.899902, 198300],
            ["10-12-2013", 6354.700195, 6362.25, 6307.549805, 6332.850098, 242400],
            ["11-12-2013", 6307.200195, 6326.600098, 6280.25, 6307.899902, 148200],
            ["12-12-2013", 6276.75, 6286.850098, 6230.549805, 6237.049805, 148500],
            ["13-12-2013", 6201.299805, 6208.600098, 6161.399902, 6168.399902, 169700],
            ["16-12-2013", 6168.350098, 6183.25, 6146.049805, 6154.700195, 146500],
            ["17-12-2013", 6178.200195, 6190.549805, 6133, 6139.049805, 154200],
            ["18-12-2013", 6129.950195, 6236, 6129.950195, 6217.149902, 243300],
            ["19-12-2013", 6253.899902, 6263.75, 6150.700195, 6166.649902, 437000],
            ["20-12-2013", 6179.950195, 6284.5, 6170.350098, 6274.25, 172000],
            ["23-12-2013", 6267.200195, 6317.5, 6266.950195, 6284.5, 131200],
            ["24-12-2013", 6296.450195, 6301.5, 6262, 6268.399902, 107600],
            ["26-12-2013", 6270.100098, 6302.75, 6259.450195, 6278.899902, 182300],
            ["27-12-2013", 6292.799805, 6324.899902, 6289.399902, 6313.799805, 96900],
            ["30-12-2013", 6336.399902, 6344.049805, 6273.149902, 6291.100098, 101300],
            ["31-12-2013", 6307.350098, 6317.299805, 6287.299805, 6304, 103400],
            ["02-01-2014", 6301.25, 6358.299805, 6211.299805, 6221.149902, 158100],
            ["03-01-2014", 6194.549805, 6221.700195, 6171.25, 6211.149902, 139000],
            ["06-01-2014", 6220.850098, 6224.700195, 6170.25, 6191.450195, 118300],
            ["07-01-2014", 6203.899902, 6221.5, 6144.75, 6162.25, 138600],
            ["08-01-2014", 6178.049805, 6192.100098, 6160.350098, 6174.600098, 146900],
            ["09-01-2014", 6181.700195, 6188.049805, 6148.25, 6168.350098, 150100],
            ["10-01-2014", 6178.850098, 6239.100098, 6139.600098, 6171.450195, 159900],
            ["13-01-2014", 6189.549805, 6288.200195, 6189.549805, 6272.75, 135000],
            ["14-01-2014", 6260.25, 6280.350098, 6234.149902, 6241.850098, 110200],
            ["15-01-2014", 6265.950195, 6325.200195, 6265.299805, 6320.899902, 145900],
            ["16-01-2014", 6341.350098, 6346.5, 6299.850098, 6318.899902, 153400],
            ["17-01-2014", 6306.25, 6327.100098, 6246.350098, 6261.649902, 167700],
            ["20-01-2014", 6261.75, 6307.450195, 6243.350098, 6303.950195, 122100],
            ["21-01-2014", 6320.149902, 6330.299805, 6297.899902, 6313.799805, 141700],
            ["22-01-2014", 6309.049805, 6349.950195, 6287.450195, 6338.950195, 137500],
            ["23-01-2014", 6325.950195, 6355.600098, 6316.399902, 6345.649902, 120100],
            ["24-01-2014", 6301.649902, 6331.450195, 6263.899902, 6266.75, 160300],
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

print(df)
"""