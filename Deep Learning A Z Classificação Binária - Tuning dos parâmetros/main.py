#### python3 -m venv venv | source venv/bin/activate |pip install scikeras numpy pandas matplotlib scikit-learn tensorflow jupyter
import pandas as pd
import tensorflow as tf
import sklearn
import scikeras


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn.model_selection import GridSearchCV

# Importa os dados para treinamento
X = pd.read_csv("entradas_breast.csv")
Y = pd.read_csv("saidas_breast.csv")

def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    K.clear_session() # Limpa a sessao
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)), # Cama de entrada define a quantidade de neuronios, no caso 30 igual visto em X_train
        tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 10)

#Parametros a serem testados
parametros = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8]
}

# CV = Quantidade de vezes que vai ser feito o treinamento, os dados serao divididos em 10 partes
grid_search = GridSearchCV(estimator = rede_neural, param_grid = parametros, scoring = 'accuracy', cv = 5)

grid_search = grid_search.fit(X, Y)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print("Melhores parametros: {melhores_parametros} | Melhor precisao: {melhor_precisao}".format(melhores_parametros=melhores_parametros, melhor_precisao=melhor_precisao))
