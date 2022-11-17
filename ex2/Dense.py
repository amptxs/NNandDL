import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('data/Khabarovsk_weather_15102020_15102012.csv', sep=';', encoding='windows-1251')
df = df.dropna()

minMaxScaler = MinMaxScaler()
X = df.drop(['T', 'LocalTime', 'DD'], axis=1)
y = df['T']
X = minMaxScaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressorModel = Sequential()
regressorModel.add(Dense(15, activation='relu', input_dim=X_train.shape[1]))
regressorModel.add(Dense(1))
regressorModel.summary()
regressorModel.compile(optimizer='adam', loss='mse',
                       metrics='mae')

regressorHistory = regressorModel.fit(X_train, y_train,
                                      batch_size=100, epochs=20)

pd.DataFrame(regressorHistory.history).plot(figsize=(8, 5))
plt.show()

y_pred = regressorModel.predict(X_test)

print('R2 Score = ', round(r2_score(y_true=y_test, y_pred=y_pred), 2))
print('MAE = ', round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2))
