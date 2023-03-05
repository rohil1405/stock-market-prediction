import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end='2021-12-31'

st.title("Stock Tread Analysis")

user_input=st.text_input("Enter Stock Ticker",'AAPL')
df=data.DataReader('AAPL','yahoo',start, end)

st.subheader("Data From 2010-2019")
write(df.describe())