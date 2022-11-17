import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense


# игнорируем предупреждения
def warn(*args, **kwargs):
    pass


warnings.warn = warn


# функция конвертирования строки в float
def toNum(s):
    try:
        return float(s.replace(',', '.'))
    except:
        return s


# считывание и предобработка датасета
df = pd.read_csv('data/frac.csv', sep=';', encoding='windows-1251')
for col in df.columns:
    df[col] = df[col].apply(toNum)

df = df.fillna(df.median())

# деление выборки на данные по которым будет выполняться обучение и прогнозирование
X = df[['well_index', 'year', 'h', 'porosity', 'permeability_avg', 'init_oil_saturation_factor', 'aps_avg', 'sandiness',
        'number_of_permeable_intervals', 'recoverables', 'remaining_recoverables', 'base_qliq1', 'base_qliq3',
        'cumulative_fluid_production',
        'bottomhole_pressure6', 'kprod_current', 'previous_grp_result_qliq3', 'rate_liq_avg_1000', 'liq_sum_1000',
        'receptivity_avg_nagn1',
        'bottomhole_pressure_nagn1']]
y = df['result_qoil3']

X = X.drop(['year', 'well_index'], axis=1)

# деление выборки на выборки для обучения и тестов
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# обучение регрессора keras
kerasModel = Sequential()
kerasModel.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
kerasModel.add(Dense(1))
kerasModel.compile(optimizer='adam', loss='mse', metrics='mae')
kerasModel.summary()
kerasModelHistory = kerasModel.fit(X_train, y_train, batch_size=100, epochs=100)

# график обучения для keras
pd.DataFrame(kerasModelHistory.history).plot(figsize=(8, 5))
plt.show()

y_pred = kerasModel.predict(X_test)
print(f'MAE keras =  {round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 2)}')
print(f'R2 Score = {round(r2_score(y_true=y_test, y_pred=y_pred), 2)}')
