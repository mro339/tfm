import os # Para manejar variables de entorno. Aquí se usa para obtener el ID del cliente desde una variable de entorno.
import flwr as fl
import tensorflow as tf
import numpy as np

# ---------------------------
# Sección de cargar datos MNIST
# ---------------------------


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #Descargar y cargar el conjunto de datos MNIST, que contiene imágenes de dígitos escritos a mano

# Normalizar los datos a [0, 1], ya que los valores de píxeles van de 0 a 255, y así facilitar el entrenamiento del modelo
x_train = x_train / 255.0
x_test = x_test / 255.0

# Redimensionar para red neuronal, agregando canal de color (1 para escala de grises), esto significa que cada imagen tendrá dimensiones 28x28x1, por lo cual el modelo espera esa forma de entrada
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#Para hacer una distribución de datos non-IDD, hacemos los siguientes pasos.
#1. Ordenamos los datos por etiqueta
# Ordenamos los datos por su etiqueta (de 0 a 9)
indices_ordenados = np.argsort(y_train)
x_train_sorted = x_train[indices_ordenados]
y_train_sorted = y_train[indices_ordenados]


client_id = int(os.environ.get("CLIENT_ID", "1")) #Obtener el ID del cliente desde una variable de entorno, por defecto es 1 si no se especifica
num_clients = int(os.environ.get("TOTAL_CLIENTS", "2")) #Número total de clientes (debe coincidir con el número de instancias de cliente que se ejecutan)

#A modo de identficarlo en los logs.
print(f"🔢 Cliente ID: {client_id} de {num_clients}")

partition_size = len(x_train) // num_clients #Tamaño de la partición de datos para cada cliente. // significa división entera

start = (client_id - 1) * partition_size #Índice de inicio de la partición de datos para este cliente
end = start + partition_size #Índice de fin de la partición de datos para este cliente

x_train_c = x_train_sorted[start:end] #Datos de entrenamiento para este cliente
y_train_c = y_train_sorted[start:end] #Etiquetas de entrenamiento para este cliente

print(f"📡 Cliente {client_id} arrancado.")
print(f"   -> Etiquetas únicas en este cliente: {np.unique(y_train_c)}")
print(f"   -> Total imágenes: {len(x_train_c)}")

# ---------------------------
# Sección del modelo Keras simple
# ---------------------------
def build_model(): #Construir un modelo de red neuronal convolucional simple para clasificar las imágenes de MNIST
    model = tf.keras.Sequential([ #Modelo secuencial apilando capas linealmente. signifca que la salida de una capa es la entrada de la siguiente
        tf.keras.layers.Conv2D(8, 3, activation="relu", input_shape=(28, 28, 1)), #Capa convolucional con 32 filtros de tamaño 3x3 y función de activación ReLU
        tf.keras.layers.MaxPooling2D(), #Capa de pooling para reducir la dimensionalidad espacial
        tf.keras.layers.Flatten(), #Aplanar la salida 2D a 1D para conectarla a capas densas
        tf.keras.layers.Dense(16, activation="relu"), #Capa densa (fully connected) con 64 neuronas y función de activación ReLU
        tf.keras.layers.Dense(10, activation="softmax"), #Capa de salida con 10 neuronas (una por cada clase de dígitos) y función de activación softmax para clasificación multiclase
    ])

    model.compile( #Compilar el modelo con el optimizador Adam, la función de pérdida de entropía cruzada categórica y la métrica de precisión
        optimizer="adam", #Optimizador Adam, que es eficiente y ampliamente utilizado para entrenar redes neuronales
        loss="sparse_categorical_crossentropy", #Función de pérdida para clasificación multiclase con etiquetas enteras
        metrics=["accuracy"], #Métrica para evaluar el rendimiento del modelo durante el entrenamiento y la evaluación
    )

    return model


model = build_model()


# ---------------------------
# Sección del cliente Flower
# ---------------------------
class FlowerClient(fl.client.NumPyClient): #Definir un cliente Flower que implementa los métodos necesarios para el entrenamiento y evaluación federados

    def get_parameters(self, config=None):
        return model.get_weights()

    def fit(self, parameters, config=None):
        model.set_weights(parameters)
        model.fit(x_train_c, y_train_c, epochs=1, batch_size=32, verbose=0)
        return model.get_weights(), len(x_train_c), {}

    def evaluate(self, parameters, config=None):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=2)
        return loss, len(x_test), {"accuracy": acc}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="server:8080", client=FlowerClient())
