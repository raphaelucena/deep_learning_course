#### python3 -m venv venv | source venv/bin/activate |pip install scikeras numpy pandas matplotlib scikit-learn tensorflow jupyter
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder


# Dados
base = pd.read_csv('iris.csv')

# Obtem os dados X e Y
X = base.iloc[:, 0:4].values
Y = base.iloc[:, 4].values

# Codifica os dados
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Transforma os dados em categorias
Y = np_utils.to_categorical(Y)

# Divide os dados
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size=0.25)

# Cria o modelo
rede_neural = Sequential()
rede_neural.add(InputLayer(shape=(4,)))
rede_neural.add(Dense(units=4, activation='relu')) 
rede_neural.add(Dense(units=4, activation='relu'))
rede_neural.add(Dense(units=3, activation='softmax'))

print(rede_neural.summary())

rede_neural.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

rede_neural.fit(X_treinamento, Y_treinamento, batch_size=10, epochs=1000)

rede_neural.evaluate(X_teste, Y_teste)

previsoes = rede_neural.predict(X_teste)

previsoes = (previsoes > 0.5)


Y_teste2 = [np.argmax(t) for t in Y_teste]
previsoes = [np.argmax(t) for t in previsoes]

# print(Y_teste2)
# print(previsoes)

print(accuracy_score(Y_teste2, previsoes))

print(confusion_matrix(Y_teste2, previsoes))
