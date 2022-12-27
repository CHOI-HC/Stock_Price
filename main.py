#%%
import pandas as pd
import numpy as np

from tensorflow.python.client import device_lib
import tensorflow as tf

# 딥러닝 디폴트
from keras import models, Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import os
import warnings
warnings.filterwarnings('ignore')

# %%
# 데이터
file = 'data/stock.csv'
df = pd.read_csv(file)
df.head()
# %%
# 시계열 데이터는 시간(Date)을 index로 설정
# shape
print(f'df shape: {df.shape}')
# %%
# info
df.info()
# %%
# isnull
df.isnull().sum()
# %%
# 컬럼명 소문자로 변경, date컬럼의 데이터 타입을 datetype으로 변경
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])
df.info()
# %%
# date 컬럼을 index로 설정
df = df.set_index('date')
df.head()
# %%
# 시계열 데이터 확인
plt.figure(figsize=(12,5))

#%%
# 종가 추이 확인(index가 x축으로 자동으로 지정)
plt.plot(df['close'], label='close', color='r')
plt.show()

#%%
# df 전체 plot
df.plot()

#%%
# df 전체의 컬럼별 plot
df.plot(subplots=True)

#%%
# lineplot
px.line(df['open'], title='CocaCola stock Open Price')

#%%
# 시가, 상한가, 하한가, 종가 시각화
fig = make_subplots(rows=4, cols=1, subplot_titles=("open", "high", "low", "close"))
fig.add_trace(go.Line(x=df.index, y=df["open"], name="open"), row=1, col=1)
fig.add_trace(go.Line(x=df.index, y=df["high"], name="high"), row=2, col=1)
fig.add_trace(go.Line(x=df.index, y=df["low"], name="low"), row=3, col=1)
fig.add_trace(go.Line(x=df.index, y=df["close"], name="close"), row=4, col=1)
fig.update_layout(title=dict(text="CocaCola Stock"))
fig.show()

#%%
fig = go.Figure(data=[go.Candlestick(x=df[:30].index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
fig.show()

plt.plot(df['close'], label='close', color='r')
plt.show()

px.line(df['close'], title='CocaCola Stock Close Price')

fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df['close'], name='close'))
fig.update_layout(title=dict(text="CocaCola Stock Close Price"))
fig.show()

#%%
#### Moving Average
# 추이 확인 > 이동평균
df = df[['close']]
df.head(20)

#%%
# 300, 500, 700, 900
mas = [300, 500, 700, 900]
for ma in mas:
    df[f'ma{ma}days'] = df['close'].rolling(ma).mean()

#%%
fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df['close'], name='closePrice'))
fig.add_trace(go.Line(x=df.index, y=df['ma300days'], name='300days'))
fig.add_trace(go.Line(x=df.index, y=df['ma500days'], name='500days'))
fig.add_trace(go.Line(x=df.index, y=df['ma700days'], name='700days'))
fig.add_trace(go.Line(x=df.index, y=df['ma900days'], name='900days'))
fig.update_layout(title=dict(text="CocaCola Stock, Close Price Moving_Average"))
fig.show()

#%%
#### Pct_change
#전일 대비 종가 등락폭 확인
df['pct_change'] = df['close'].pct_change()
df.head(30)

#%%
px.line(df['pct_change'], title='Close Price change per day')

#%%
# risk histogram
px.histogram(x=df['pct_change'], title='Close Price Change Distribution')

#%%
pct_std = df['pct_change'].std()
pct_avg = df['pct_change'].mean()

fig = go.Figure()
fig.add_trace(go.Line(x=[pct_avg], y=[pct_std]))

#%%
#### 스케일링
df = df[['close']]

mms = MinMaxScaler()
df['close'] = mms.fit_transform(df[['close']])
df.head()

#%%
## 딥러닝
### 데이터 분할
# 전체 데이터의 85%를 train dataset, 나머지 15%를 test dataset으로 지정
# LSTM: t-1기의 시점이 t기의 시점에 영향을 주는 모델
# 따라서 dataset을 1~60, 2~61, ..., n~(n+60)의 형식으로 맞물리게 구성
# y dataset은 X dataset 한 묶음의 다음 날의 종가를 사용(61, 62, ..., (n+61))
# 60일 단위로 구성

dataset = df['close'].values

X = []
y = []
for i in range(len(dataset)-60):
    X.append(dataset[i: i+60])
    y.append(dataset[i+60])
# %%
# X, y dataset 길이 확인
print(f'lenght of X: {len(X)}')
print(f'lenght of y: {len(y)}')

#%%
# 데이터 분할(tran, test)
X_train = np.array(X[:int(len(X)*0.85)+1])
y_train = np.array(y[:int(len(X)*0.85)+1])
X_test = np.array(X[int(len(X)*0.85)+1:])
y_test = np.array(y[int(len(X)*0.85)+1:])

print(f'length of X_train: {len(X_train)}')
print(f'length of X_test: {len(X_test)}')
print(f'length of y_train: {len(y_train)}')
print(f'length of y_test: {len(y_test)}')

#%%
### LSTM
# input_shqpe > 1)batch_size, 2) 시점(n), 3) feature 수
# 1)batch_size: 한 번에 학습하는 데이터 수(최적의 값을 알 수 없기 때문에 적지 않음(default:None))
# 2)시점(n): 몇 개의 데이터를 가지고 n+1기의 시점을 예측하는지
# 3)feature 수: 종가만을 가지고 예측하므로 현재의 경우 1
# input train data는 (60,)의 데이터가 총 12,803개이므로 이를 (12,803, 60, 1)로 변형

X_train = X_train.reshape(-1, 60, 1)
print(f'input data shape: {X_train.shape}')

# %%
# LSTM 특성 상 여러 개의 입력을 받고, 각 노드들이 다음 노드에 영향을 주면서 옆으로 밀려 하나의 값을 출력
# LSTM을 두 개 이상으로 쌓는 경우 이전 출력값과 동일한 노드 수가 필요
# 따라서 첫 LSTM의 각 노드마다 출력값이 나올 수 있도록 return_sequences=True를 설정

model = Sequential()
model.add(LSTM(256, input_shape=(60,1), return_sequences=True))
model.add(LSTM(128, activation='elu', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()
history = model.fit(X_train, y_train, epochs=10, verbose=1)

# %%
# loss lineplot
sns.lineplot(history.history['loss'])
# %%
pred = model.predict(X_test)
pred_inv = mms.inverse_transform(pred)
y_test_inv = mms.inverse_transform(y_test.reshape(-1,1))

#%%
pred_df = pd.DataFrame({'close': y_test_inv.reshape(-1,), 'pred':pred_inv.reshape(-1,)})
pred_df.index = df.index[len(X_train)-len(df)+60:]
pred_df.head()

#%%
fig = go.Figure()
for col in pred_df.columns:
    fig.add_trace(go.Line(x=pred_df.index, y=pred_df[col]))
fig.update_layout(title='Close, Pred')
fig.show()
# %%
from sklearn.metrics import mean_squared_error
mse = mean_squared_error
mse(y_test_inv, pred_inv)

# %%
