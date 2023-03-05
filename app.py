from pickletools import optimize
from tkinter import Label
import numpy as np
import pandas as pd
import pandas_datareader as dr
import pandas_datareader as data
import matplotlib.pyplot as plt
import streamlit as st
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf

start='2010-01-01'
end='2021-12-31'

df = data.DataReader('BABA','yahoo',start, end)
df.head()

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock: ', 'ITC')
df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data:-')
st.write(df.describe())

st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'black')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200, 'g')

st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 300MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma300 = df.Close.rolling(300).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
plt.plot(ma300, 'g')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 400ma')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma300 = df.Close.rolling(300).mean()
ma400 = df.Close.rolling(400).mean()
plt.plot(ma100, 'y')
plt.plot(ma200, 'r')
plt.plot(ma300, 'g')
plt.plot(ma400, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)
data_training = data_training.head()
data_testing = data_testing.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):

    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.shape
y_train = y_train.shape

model = load_model('SGP.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):

    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
tf.random.set_seed(42)

# Create some regression data
X_regression = np.expand_dims(np.arange(0, 1000, 5), axis=1)
y_regression = np.expand_dims(np.arange(100, 1100, 5), axis=1)

# Split it into training and test sets
X_reg_train = X_regression[:150, :]
X_reg_test = X_regression[150:, :]

y_reg_train = y_regression[:150, :]
y_reg_test = y_regression[150:, :]

tf.random.set_seed(42)

model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])
#y_predicted = model.predict(x_test)
#y_predicted = scaler.inverse_transform(y_predicted)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
# y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma100,'v', label = 'Predictions Price')
plt.plot(ma200, 'y', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

colors = ["red", "white", "blue"]
colors.insert(2, "yellow")


