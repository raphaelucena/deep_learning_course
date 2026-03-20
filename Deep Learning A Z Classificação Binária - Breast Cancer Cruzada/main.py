#### python3 -m venv venv | source venv/bin/activate |pip install scikeras numpy pandas matplotlib scikit-learn tensorflow jupyter
import pandas as pd
import tensorflow as tf
import sklearn
import scikeras


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K

# Importa os dados para treinamento
X = pd.read_csv("entradas_breast.csv")
Y = pd.read_csv("saidas_breast.csv")

def criar_rede():
    K.clear_session() # Limpa a sessao
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape = (30,)), # Cama de entrada define a quantidade de neuronios, no caso 30 igual visto em X_train
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
        tf.keras.layers.Dropout(0.2), # Dropout para evitar overfitting, Evite colocar na primeira camada, pois pode zerar atributos importantes
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return rede_neural

rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 10)


# CV = Quantidade de vezes que vai ser feito o treinamento, os dados serao divididos em 10 partes
resultado = cross_val_score(estimator=rede_neural, X = X, y = Y, cv = 10, scoring="accuracy")

print(resultado)

# Media dos resultados
media = resultado.mean()
desvio_padrao = resultado.std()

# Qualidade do modelo, quanto menor o desvio padrao, melhor
print("Media: {media} | Desvio padrao: {desvio_padrao}".format(media=media, desvio_padrao=desvio_padrao))