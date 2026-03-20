
import tensorflow as tf
import numpy as np

classificador = tf.keras.models.load_model('breast_cancer.keras')

novo = np.array([[ 15.80,  8.34,   118,   900,  0.10,  0.26,  0.08, 0.134, 0.178,  0.20,
     0.05,  1098,  0.87,  4500, 145.2, 0.005,  0.04,  0.05, 0.015,  0.03,
    0.007, 23.15, 16.64, 178.5,  2018,  0.14, 0.185,  0.84,   158, 0.363]])


previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

if previsao:
    print("Maligno")
else:
    print("Benigno")