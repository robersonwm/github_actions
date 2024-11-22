# desarrollar un modelo que cada vez que hagamos push a un repositorio de Git, se actualice y haga una serie de validaciones y los resultados del modelo 
# se queden almacenados dentro de Git, de manera que sea muy fácil comparar el desempeño de una versión con el de otra versión


# modelo de regresión simple usando TensorFlow, que incluye la definición, entrenamiento, predicción y evaluación del modelo. También genera gráficos y guarda métricas en un archivo.
import tensorflow as tf                 # Biblioteca principal para construir y entrenar modelos de aprendizaje automático.
import numpy as np                      # Usado para manejar datos numéricos y vectores.
import matplotlib.pyplot as plt         # Utilizado para graficar los datos y las predicciones.


# Función para graficar predicciones
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  plt.figure(figsize=(6, 5))

  # Grafica los datos de entrenamiento (train_data y train_labels)
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  
  # grafica datos de prueba (test_data y test_labels)
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  
  # grafica las predicciones del modelo (predictions).
  plt.scatter(test_data, predictions, c="r", label="Predictions")

  # Configura detalles como leyendas, títulos y rejillas para mejorar la visualización.
  plt.legend(shadow='True')
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)

  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)

  # Guarda el gráfico en un archivo PNG llamado model_results.png.
  plt.savefig('model_results.png', dpi=120)


# Funciones para calcular métricas de error
# Calcula el Error Absoluto Medio (MAE) entre las etiquetas reales (y_test) y las predicciones (y_pred).
def mae(y_test, y_pred):                                    
  return tf.metrics.mean_absolute_error(y_test, y_pred)
  
# Calcula el Error Cuadrático Medio (MSE) entre las etiquetas reales (y_test) y las predicciones (y_pred).
def mse(y_test, y_pred):
  return tf.metrics.mean_squared_error(y_test, y_pred)      

# Imprime la versión de TensorFlow
print(tf.__version__)

# Genera un rango de números con un paso de 4 para usar como datos de entrada (X) y etiquetas (y).
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Divide los datos en conjuntos de entrenamiento (primeros 25 puntos) y prueba (resto).
N = 25
X_train = X[:N].reshape(-1, 1)  # Convertir a (N, 1)
y_train = y[:N]

X_test = X[N:].reshape(-1, 1)   # Convertir a (N, 1)
y_test = y[N:]


# Construccion del modelo
# Fija la semilla aleatoria para reproducibilidad.
tf.random.set_seed(1989)

# Define un modelo secuencial con dos capas densas, ambas con una sola neurona.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),  # Especifica la forma de entrada
    tf.keras.layers.Dense(1)
])

# Compila el modelo
model.compile(loss = tf.keras.losses.MeanAbsoluteError(),
              optimizer=tf.keras.optimizers.SGD(),      # Descenso por gradiente estocástico (SGD)
              metrics = ['mae'])                          # Métrica de evaluación (MAE).

# Entrena el modelo con los datos de entrenamiento durante 100 épocas.
model.fit(X_train, y_train, epochs=100)

# Genera predicciones para los datos de prueba (X_test).
y_preds = model.predict(X_test)


# Graficar resultados
# Genera y guarda un gráfico que compara los datos reales y las predicciones.
plot_predictions(train_data=X_train, train_labels=y_train,
                 test_data=X_test, test_labels=y_test, predictions=y_preds)

# Calcula las métricas de error (MAE y MSE) entre los valores reales (y_test) y las predicciones (y_preds).
mae_1 = np.round(float(tf.keras.metrics.mean_absolute_error(y_test, y_preds.squeeze()).numpy()), 2)
mse_1 = np.round(float(tf.keras.metrics.mean_squared_error(y_test, y_preds.squeeze()).numpy()), 2)

# Muestra las métricas calculadas.
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# Guarda las métricas en un archivo de texto llamado metrics.txt.
#with open('metrics.txt', 'w') as outfile:
#    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
