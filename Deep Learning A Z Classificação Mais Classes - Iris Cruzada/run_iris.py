import tensorflow as tf
import numpy as np

# Carrega modelo treinado no Iris
classificador = tf.keras.models.load_model('iris_best_model.h5')

# Novo dado (exemplo Iris)
# [sepal_length, sepal_width, petal_length, petal_width]
novo = np.array([[5.1, 3.5, 1.4, 0.2]])

# Previsão
previsao = classificador.predict(novo)

# Pega a classe com maior probabilidade
classe = np.argmax(previsao)

# Mapeamento das classes
classes = ['setosa', 'versicolor', 'virginica']

print("Previsão:", classes[classe])
print("Probabilidades:", previsao)