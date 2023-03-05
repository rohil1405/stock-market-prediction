import numpy as np
import pandas as pd
import pandas_datareader as dr
import pandas_datareader as data
 
df=dr.data.get_data_yahoo("ibm",start='2018-09-27',end='2018-10-23')
start='2010-01-01'
end='2021-12-31'

df=data.DataReader('TSLA','yahoo',start, end)
df.head()