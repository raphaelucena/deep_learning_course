import pandas as pd

import sklearn 
from sklearn.model_selection import train_test_split

# Verifica a versão do panda
print('Pandas version: {pd.__version__}'.format(pd=pd))

# Carrega os arquivos previsores *** Dados usados para treinar a IA dados de amostras
X = pd.read_csv('entradas_breast.csv')

# Carrega os arquivos de classe *** Dados usados para treinar a IA dados de resultados
Y = pd.read_csv('saidas_breast.csv')

print(X)
print(Y)

# Versão do SKLearn
print('Sklearn version: {sklearn.__version__}'.format(sklearn=sklearn))

# Base de treinamento e teste Define que 25% dos dados serao usados para teste e 75% para treinamento
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print("Dados de X treinamento: {X_train.shape}".format(X_train=X_train))
print("Dados para X test: {X_test.shape}".format(X_test=X_test))
print("Dados para Y_train: {Y_train.shape}".format(Y_train=Y_train))
print("Dados para Y_test: {Y_test.shape}".format(Y_test=Y_test))

# Estrutora da rede neural
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Versão do tensorflow
# print('Tensorflow version: {tf.__version__}'.format(tf=tf))

# rede_neural = Sequential([
#     tf.keras.layers.InputLayer(shape = (30,)), # Cama de entrada define a quantidade de neuronios, no caso 30 igual visto em X_train
#     tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
#     tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
# ]#)

rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape = (30,)), # Cama de entrada define a quantidade de neuronios, no caso 30 igual visto em X_train
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
    tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'), # Rede neural densa, cama oculta para somar a quantidade de units = (<entradas> + <saidas>) / 2 | (30 + 2) / 2 = 16
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
])

# Estrutura da rede neural
rede_neural.summary()

# Reconfigura o Adam
otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)

# Compilacao da rede neural
rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
# rede_neural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# Treinamento da rede neural
rede_neural.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# Previsoes
previsoes = rede_neural.predict(X_test)

# Conversao das previsoes em 0 ou 1
previsoes = (previsoes > 0.5)

# print("Previsoes: {previsoes}".format(previsoes = previsoes))

# Comparacação das privisoes com os resultados reais
from sklearn.metrics import confusion_matrix, accuracy_score

# 
precisao = accuracy_score(Y_test, previsoes)
matriz = confusion_matrix(Y_test, previsoes)

# Percentagem de acerto
print("Precisao: {precisao}".format(precisao = precisao))

# Matriz de confusao, quantidade de acertos e erros
print("Matriz: {matriz}".format(matriz = matriz))

# Acertos e erros, para calucar os erros e acertos, somar em X
print("Acertos: {matriz}".format(matriz = matriz[0][0] + matriz[1][1]))
print("Erros: {matriz}".format(matriz = matriz[0][1] + matriz[1][0]))

# Percentual de acerto e erro
rede_neural.evaluate(X_test, Y_test)

# Obtem os pesos da rede neural entrada para primeira camada oculta
pesos0 = rede_neural.layers[0].get_weights()

# Primeira posição é os neuronios, segunda os biases
print(len(pesos0[0]), len(pesos0[1]))

# Obtem os pesos da rede neural primeira para segunda camada oculta
pesos1 = rede_neural.layers[1].get_weights()
print(len(pesos1[0]), len(pesos1[1]))

# obtem os pesos da rede neural segunda para a camada de saida
pesos2 = rede_neural.layers[2].get_weights()
print(len(pesos2[0]), len(pesos2[1]))

