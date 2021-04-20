# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


rcParams['figure.figsize'] = 20, 10

scaler = MinMaxScaler(feature_range=(0, 1))

df = yf.download("TSLA", start="2018-01-01", end="2020-12-31", interval="1d")
df.head()
df.tail()
df.describe()
df.columns

# plot tsla closing stock price
plt.figure(figsize=(15, 7))
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(df['Close'], label='Close Price history')

# plot tsla closing stock price along with moving average
df["MA1"] = df['Close'].rolling(window=50).mean()
df["MA2"] = df['Close'].rolling(window=200).mean()
plt.figure(figsize=(15, 7))
plt.plot(df['MA1'], 'g--', label="MA1")
plt.plot(df['MA2'], 'r--', label="MA2")
plt.plot(df['Close'], label="Close")
plt.legend()

# plot tsla closing stock price along with Bollinger Bands
df['Middle Band'] = df['Close'].rolling(window=20).mean()
df['Upper Band'] = df['Close'].rolling(window=20).mean() + df['Close'].rolling(window=20).std()*2
df['Lower Band'] = df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=20).std()*2
plt.figure(figsize=(15, 7))
plt.plot(df['Upper Band'].iloc[-200:], 'g--', label="Upper")
plt.plot(df['Middle Band'].iloc[-200:], 'r--', label="Middle")
plt.plot(df['Lower Band'].iloc[-200:], 'y--', label="Lower")
plt.plot(df['Close'], label="Close")
plt.legend()

# pearson correlation coefficient
corr = df.corr(method='pearson')
corr

# set a new dataframe containing just date range and closing price column
new_df = pd.DataFrame(df, columns=['Close'])
new_df = new_df.reset_index()
new_df.head()

# checked to see if new dataframe had any missing values
new_df.isna().values.any()

# created train and test set and linear regression model
train, test = train_test_split(new_df, test_size=.3)

x_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']
model = LinearRegression()
model.fit(x_train, y_train)
print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
print('Intercept: ', model.intercept_)

# graphed linear regression model
plt.figure(1, figsize=(15, 7))
plt.title('Linear Regression | Price vs Time')
plt.scatter(x_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(x_train, model.predict(x_train), color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
