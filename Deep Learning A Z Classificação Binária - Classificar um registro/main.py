#### python3 -m venv venv | source venv/bin/activate |pip install scikeras numpy pandas matplotlib scikit-learn tensorflow jupyter
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout

# Importa os dados para treinamento
X = pd.read_csv("entradas_breast.csv")
Y = pd.read_csv("saidas_breast.csv")

classificador = Sequential()
classificador.add(InputLayer(shape = (30,)))
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classificador.fit(X, Y, batch_size = 10, epochs = 100)

score = classificador.evaluate(X, Y)
print(score)

novo = np.array([[ 15.80,  8.34,   118,   900,  0.10,  0.26,  0.08, 0.134, 0.178,  0.20,
                    0.05,  1098,  0.87,  4500, 145.2, 0.005,  0.04,  0.05, 0.015,  0.03,
                   0.007, 23.15, 16.64, 178.5,  2018,  0.14, 0.185,  0.84,   158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

if previsao:
    print("Maligno")
else:
    print("Benigno")


classificador.save('breast_cancer.keras')