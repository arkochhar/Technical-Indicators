# Technical-Indicators
Collection of Technical Indicators used in Stock Market Analysis. Currently there are following indications in the project;

1. Exponential Moving Average (EMA)
2. Average True Range (ATR)
3. SuperTrend
3. Moving Average Convergence Divergence (MACD)

Feel free to contribute to the list of indicators in the project. Following are the areas where contributions will be useful;

1. Additional technical indicators to the list
2. Optimizations to the existing algorithms
3. Help create as pip package

Goal is to make this publically available project where people can contribute to indicators used on technical analysis for stock markets.

import pandas as pd

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
    
- Date must be present as a Pandas DataFrame with ['date', 'open', 'high', 'low', 'close', 'volume'] as columns
df = pd.DataFrame(data["data"]["candles"], columns=['date', 'open', 'high', 'low', 'close', 'volume'])

- Columns as added by each function specific to their computations
EMA(df, 'close', 'ema_5', 5)
ATR(df, 14)
SuperTrend(df, 10, 3)
MACD(df)
