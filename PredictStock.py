# library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# data 불러오기
data = pd.read_csv("LPL (1).csv")
data.tail()
# data 전처리
data['Date'] = pd.to_datetime(data['Date'])
split_date = datetime.datetime(2018, 1, 1)
training_data = data[data['Date'] < split_date].copy()
test_data = data[data['Date'] >= split_date].copy()
#train 데이터 셋 설정
training_data = training_data.drop(['Date', 'Adj Close'], axis=1)
past_60_days = training_data.tail(60)
#리스케일링을 위한 기준 데이터셋 설정
arr_train = training_data.to_numpy()
y_fit = []
for row in arr_train:
    y_fit.append(row[0])
y_fit = np.array(y_fit)
#정규화를 위한 스케일링
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

x_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#LSTM 모델 설계
regression = Sequential()
regression.add(LSTM(units=50, activation="relu",
                    return_sequences=True, input_shape=(x_train.shape[1], 5)))
regression.add(Dropout(0.2))

regression.add(LSTM(units=60, activation="relu", return_sequences=True))
regression.add(Dropout(0.3))

regression.add(LSTM(units=80, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=120, activation="relu"))
regression.add(Dropout(0.5))

regression.add(Dense(units=1))

regression.compile(optimizer='adam', loss='mean_squared_error')
regression.fit(x_train, y_train, epochs=250, batch_size=32)

# RNN 이용하여 학습시키기
df = past_60_days.append(test_data, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
inputs = scaler.fit_transform(df)

x_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# 리스케일링
y_pred = regression.predict(x_test)

y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_fit = y_fit.reshape(-1, 1)

rescaler = MinMaxScaler()
rescaler = rescaler.fit(y_fit)
rescaled_pred = rescaler.inverse_transform(y_pred)
rescaled_test = rescaler.inverse_transform(y_test)

# Figure setting
plt.figure(figsize=(14, 5))
plt.plot(rescaled_test, color='red', label='Real price')
plt.plot(rescaled_pred, color='blue', label='predicted price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
plt.show()
