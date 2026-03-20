#### python3 -m venv venv | source venv/bin/activate |pip install scikeras numpy pandas matplotlib scikit-learn tensorflow jupyter
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import scikeras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as k
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline

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
def rede_neural(optimizer, loss, kernel_initializer, activation, neurons):
    model = Sequential()
    model.add(InputLayer(shape=(4,)))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units=neurons, activation=activation,  kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer=optimizer, loss = loss, metrics=['categorical_accuracy'])
    return model

# Preprocessamento para nomarlização de dados *** Adiciona clf__ ao inicio da configuração
# classificador = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', KerasClassifier(model=rede_neural, epochs=250, batch_size=10))
# ])

classificador = KerasClassifier(model=rede_neural, epochs=250, batch_size=10)

# Parametros para teste, Multiplcar as configurações para obter a media de tempo
# Ex: t = (2 x 1 x 2 x 2 x 2 x 2 x 4) * <tempo_medio_de_execução>
#     t = 128 * 1
#     t = 128
parametros = {
    'batch_size': [10],
    'epochs': [250],
    'model__optimizer': ['sgd'],
    'model__loss': ['categorical_crossentropy'],
    'model__kernel_initializer': ['glorot_uniform'],
    'model__activation': ['relu'],
    'model__neurons': [8]
}


# Configura o GridSeach
grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=10)

# Executa o GridSearch
grid_search = grid_search.fit(X_treinamento, Y_treinamento)

print("Melhor configuração:", grid_search.best_params_)
print("Melhor pontuação:", grid_search.best_score_)

# Salva o melhor modelo
melhor_modelo = grid_search.best_estimator_.model_
melhor_modelo.save('iris_best_model.h5')

#### Melhor pontuação sem o StandardScaler
# Melhor configuração: {'batch_size': 10, 'epochs': 250, 'model__activation': 'relu', 'model__kernel_initializer': 'glorot_uniform', 'model__loss': 'categorical_crossentropy', 'model__neurons': 8, 'model__optimizer': 'sgd'}
# Melhor pontuação: 0.9818181818181818
