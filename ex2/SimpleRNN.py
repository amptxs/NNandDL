import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('data/Khabarovsk_weather_15102020_15102012.csv', sep=';', encoding='windows-1251')
df = df.dropna()

minMaxScaler = MinMaxScaler()

data = df['T'].values
data = minMaxScaler.fit_transform(data.reshape(-1, 1))
data = data.flatten()
window = 10


def get_XY(data, window):
    Y_index = np.arange(window, len(data), window)
    Y = data[Y_index]
    rows_x = len(Y)
    X = data[range(window * rows_x)]
    X = np.reshape(X, (rows_x, window, 1))
    return X, Y


X, y = get_XY(data, window)
print(X.shape, y.shape)
a = int(X.shape[0] * 0.7)
b = int(X.shape[0] * 0.9)

X_train = X[:a, :]
X_val = X[a:b:]
X_test = X[b:, :]
y_train = y[:a]
y_test = y[b:]

modelSimpleRNN = Sequential()
modelSimpleRNN.add(SimpleRNN(10, activation='relu', input_shape=(10, 1)))
modelSimpleRNN.add(Dense(1, activation='linear'))

modelSimpleRNN.summary()
modelSimpleRNN.compile(optimizer='adam', loss='mse', metrics='mae')
historySimpleRNN = modelSimpleRNN.fit(X_train, y_train, epochs=10, batch_size=50)

pd.DataFrame(historySimpleRNN.history).plot(figsize=(8, 5))
plt.show()

y_pred = modelSimpleRNN.predict(X_test)
print('R2 Score = ', round(r2_score(y_true=y_test, y_pred=y_pred), 2))
print('MAE = ', round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2))
