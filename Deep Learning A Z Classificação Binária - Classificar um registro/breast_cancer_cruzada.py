import pandas as pd
import tensorflow as tf
import sklearn
import scikeras


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criar_rede():
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'he_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'he_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return rede_neural

# Pipeline com scaler, normalizador e classificador para melhorar a comprensao dos dados pela rede neural
classificador = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KerasClassifier(build_fn = criar_rede, epochs = 200, batch_size = 16))
])

resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

print("Media: {media} | Desvio padrao: {desvio_padrao}".format(media=media, desvio_padrao=desvio))