import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import mean_absolute_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import warnings


# игнорируем предупреждения
def warn(*args, **kwargs):
    pass


warnings.warn = warn

# предобработаем датасет
df = pd.read_csv('data/result_slice.csv')
df1 = df[['AGK', 'BK', 'DT', 'GGKP', 'GK', 'NKT', 'EF_b']]
df1 = df1.dropna()
df1 = df1[4 >= df1['EF_b']]

# деление выборки на данные по которым будет выполняться обучение и прогнозирование
# -1.0 чтобы значения начаи
X = df1[['AGK', 'BK', 'DT', 'GGKP', 'GK', 'NKT']]
y = df1['EF_b'] - 1.0

stdScaler = StandardScaler()
X = stdScaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifierSklearn = MLPClassifier(hidden_layer_sizes=(1, 2), activation='logistic',
                        early_stopping=True, max_iter=5000)
classifierSklearn.fit(X_train, y_train)
classifierSklearn_pred = classifierSklearn.predict(X_test)

y_train = to_categorical(y_train)
classifierKeras = Sequential()
classifierKeras.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
classifierKeras.add(Dense(4, activation='softmax'))
classifierKeras.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics='accuracy')
print(classifierKeras.summary())
classifierKerasHistory = classifierKeras.fit(X_train, y_train,
                                                  batch_size=25, epochs=25)
classifierKeras_pred = np.argmax(classifierKeras.predict(X_test), axis=1)

pd.DataFrame(classifierKerasHistory.history).plot(figsize=(8, 5))
plt.show()
print("classifierSklearn\n" + classification_report(y_true=y_test, y_pred=classifierSklearn_pred))
print("classifierKeras\n" + classification_report(y_true=y_test, y_pred=classifierKeras_pred))
