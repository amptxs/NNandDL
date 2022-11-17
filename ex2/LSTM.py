import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('data/Khabarovsk_weather_15102020_15102012.csv', sep=';', encoding='windows-1251')
df = df.dropna()

minMaxScaler = MinMaxScaler()

data = df['T'].values
data = minMaxScaler.fit_transform(data.reshape(-1, 1))
data = data.flatten()
window = 10

n_samples = int(data.shape[0] - window)

n_train_samples = int(data.shape[0] * 0.7)
n_val_samples = int(data.shape[0] * 0.15)
n_test_samples = int(n_samples - n_train_samples - n_val_samples)
X_train = np.zeros((n_train_samples, window))
y_train = np.zeros(n_train_samples)

X_test = np.zeros((n_test_samples, window))
y_test = np.zeros(n_test_samples)
for i in range(n_train_samples):
    for j in range(window):
        X_train[i, j] = data[i + j]
    y_train[i] = data[i + window]


for i in range(n_test_samples):
    for j in range(window):
        X_test[i, j] = data[n_train_samples + n_val_samples + i + j]
    y_test[i] = data[n_train_samples + n_val_samples + i + window]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

modelLSTM = Sequential()
modelLSTM.add(LSTM(10, input_shape=(window, 1)))
modelLSTM.add(Dense(1, activation='linear'))

modelLSTM.summary()
modelLSTM.compile(optimizer='adam', loss='mse', metrics='mae')
historyLSTM = modelLSTM.fit(X_train, y_train, epochs=10, batch_size=100)
y_pred = modelLSTM.predict(X_test)

pd.DataFrame(historyLSTM.history).plot(figsize=(8, 5))
plt.show()

print('R2 Score = ', round(r2_score(y_true=y_test, y_pred=y_pred), 2))
print('MAE = ', round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2))
