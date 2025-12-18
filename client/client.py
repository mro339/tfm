import os # Para manejar variables de entorno. Aquí se usa para obtener el ID del cliente desde una variable de entorno.
import flwr as fl
import tensorflow as tf
import numpy as np

"""
**Sección de cargar datos MNIST**

El dataset MNIST, consta de 70.000 imágentes de dígitos escritos a mano (0-9) en escala de grises.
Ahora cada imagen tiene un tamaó de 28x28 píxeles. Que sería una matriz de 28 filas y 28 columnas, con un valor de pixel entre 0 y 255
0 -> Negro
255 -> Blanco
y valores intermedios que representan diferentes tonalidades de gris, que representan los bordes o difuminados del dígito.

Ahora bien, el dataset MNIST está dividido en dos partes:
- Conjunto de entrenamiento: 60.000 imágenes y sus etiquetas correspondientes (el dígito que representan)
- Conjunto de prueba: 10.000 imágenes y sus etiquetas correspondientes (el dígito que representan)

Y cada parte está compuesta por:
- Imágenes (X): Una matriz de 28x28 píxeles para cada imagen.
- Etiquetas (y): Un número entero entre 0 y 9 que indica el dígito que representa la imagen.

Es por eso que al cargar el dataset con tf.keras.datasets.mnist.load_data(), obtenemos dos tuplas:
- (x_train, y_train): Conjunto de entrenamiento, compuesto por 1. Las imagenes (matríz de 28x28) representadas en un array. 2. La etiqueta correspondiente a cada imagen. (0-9).
- (x_test, y_test): Conjunto de prueba igual que lo anterior.

Más información del dataset MNIST en:
https://interactivechaos.com/es/manual/tutorial-de-deep-learning/el-dataset-mnist
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

"""
**Ahora tenemos que realizar un proceso de procesamiento de datos:**

1. Normalización de datos. Para facilitar el entrenamiento del modelo, y no hacer que valores muy altos afecten al entrenamiento.
Dado que los valores de píxeles van de 0 a 255, los normalizamos a un rango de [0, 1] dividiendo por 255.0.
2. Redimensionar los datos.
Las redes neuronales convolucinales (CNN), esperan como entradas datos de forma (conjunto de imágenes, altura, anchura, canales).
- Altura y anchura de cada imágen, 28x28 píxeles.
- Canales: 1 para escala de grises, 3 para RGB (color), 4 para RGBA (color con transparencia), etc...
. -1 el tamaño de la array que poseemos.

Por lo que simplemente redimensionamos los datos para agregar el canal de color.
y quedaría, (conjunto de imágenes, 28, 28, 1)

Hay muchos trabajos de preprocesamiento que realiza esto.
https://www.kaggle.com/code/merfarukyce/mnist-cnn-classification
https://www.kaggle.com/code/sani84/mnist-cnn
"""
#Normalización
x_train = x_train / 255.0
x_test = x_test / 255.0

#Redimensionar, añadimos canal del color.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

""" 
**Ralizamos la distribución no-IDD de los datos entre los clientes.**
Una distribucción no-IDD signfica que los datos NO se encuentran distribuidos de manera idéntica e independiente entre los clientes. 
En este caso sería, como distribuir los números de tal forma en la que un cliente tenga más números de un tipo que de otro.
Es decir, si tenemos 3 clientes, podríamos distribuirlos de la siguiente manera:
- Cliente 1: Números 0, 1, 2, 3
- Cliente 2: Números 4, 5, 6
- Cliente 3: Números 7, 8, 9 

Bueno para la implementación de esto es bastante sencilla, simplemente ordenamos los datos por sus etiquetas (números) y luego dividimos el conjunto de datos ordenados en partes iguales entre los clientes. 
Cada etiqueta y está asociada a una imagen de x, por lo que al ordenar y_train, se ordena también x_train en consecuencia.

El flujo de trabajo es el siguiente, una vez ya ordenados los datos:
1. Obtenemos el ID de cada cliente y el número total de clientes.
2. Dividimos el conjunto de datos en particiones en función al número total de clientes. (Imaginemos que tenemos 10 clientes y las 60.000 imágenes)
3. Ahora para cada cliente, calculamos su inicio y fin, que sería la partición de datos que le corresponde.
 3.1 Inicio: el id del cliente -1 * tamaño de la partición. // Si es el cliente es 1, empieza en 0, el cliente 1 empieza en 6.000 el cliente 2 empieza en 12.000...
 3.2 Fin: Inicio + tamaño de la partición.
4. En python tenemos slicing, por lo que [start:end], por lo que coge desde el inicio, hasta fin SIN INCLUIR.
 """

#Ordenamos los datos.
indices_ordenados = np.argsort(y_train)
x_train_sorted = x_train[indices_ordenados]
y_train_sorted = y_train[indices_ordenados]

#Obtenemos el ID de cada cliente y el número total de clientes, definidos en el sistema de orígenes. 
# Ver el docker-compose.yml o generate_compose.py.
client_id = int(os.environ.get("CLIENT_ID", "1"))
num_clients = int(os.environ.get("TOTAL_CLIENTS", "2"))

#Para identficarlo en los logs.
print(f" Cliente ID: {client_id} de {num_clients}")

#Tamaño de la partición de datos para cada cliente.
partition_size = len(x_train) // num_clients # '//' significa división entera

#Incii y fin de la partición.
start = (client_id - 1) * partition_size 
end = start + partition_size 

#Datos de entrenamiento para este cliente.
x_train_c = x_train_sorted[start:end] 
y_train_c = y_train_sorted[start:end]

print(f"¡¡¡¡Cliente {client_id} ha arrancado.!!!")
print(f"  -Etiquetas únicas en este cliente: {np.unique(y_train_c)}")
print(f"  -Total imágenes: {len(x_train_c)}")

"""
**Construimos el modelo de red neuronal convolucional (CNN)**
Las redes neuronales convolucionales (CNN) es un tipo de red neuronal especializado en procesar datos con una estructura cuadrática, como imágenes.
Referencia: https://www.tensorflow.org/tutorials/images/cnn
Su funcionamiento es el siguiente:
1. Capas convulacionales. A estas capas se le aplican kernels, pequeñas matrices que recorren la imagen, para extraer la características más importantes como bordes, tecturas, etc...
2. Capas de pooling, que reducen la dimensionalidad de las características extraídas, manteniendo la información más relevante.
3. Maxpooling, selecciona el valor máximo dentro cada región cubierta por el kernel.
4. Flatten, convierte la matriz, en un vector para conectarlo con las neuronas.
5. Se genera una red neuronal normal para razonar sobre las características y decidir.
"""
def build_model(): #Construir un modelo de red neuronal convolucional simple para clasificar las imágenes de MNIST
    model = tf.keras.Sequential([ #Modelo secuencial apilando capas linealmente. signifca que la salida de una capa es la entrada de la siguiente
        #Un filtro son las característica/objetivo que la red quiere aprender.
        #Kenel es la matricula de la que se va a deslizar, en este caso es 3x3.
        #La funcion de activación ReLU.
        #La entrada de datos.
        tf.keras.layers.Conv2D(8, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(), #Reduce dimensaionalidad a 2x2.
        tf.keras.layers.Flatten(), #Pasar de matriz a vector.
        tf.keras.layers.Dense(16, activation="relu"), #Capa donde se razonan
        tf.keras.layers.Dense(10, activation="softmax"), #Capa final para clasificación multiclase.
    ])

    model.compile( #Compilar el modelo con el optimizador Adam, la función de pérdida de entropía cruzada categórica y la métrica de precisión
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"], 
    )

    return model
"""
**Compilación del modelo**
Es la configuración del modelo para el entranamiento del modelo.
1. Optimizador:
El optimizador ADAM (Adaptive Moment Estimation). Es un algoritmo de optimización.
Aquí se aplica el descenso de la gradiente estocástico, para minimizar el error del modelo. Explicado en clase.
El objetivo es encontrar un mínimo local de la función de pérdida

2. Función de pérdida:
La función de pérdida de entropía cruzada categórica. Mide cuanto mal lo hizo el modelo. Para ello calcula la pérdida (loss)
Sparse significa que las etiquetas son enteros (0-9) en lugar de vectores one-hot.

3. Métrica:
La métrica de precisión para saber cuanto acertó de x imágnes.

"""
    


model = build_model()

"""
**Cliente Flower**
Estamos trabajando con Flower (FLWR), un framework de aprendizaje federado.
Recordemos que hemos utilizado una red neuronal convolucional (CNN) para clasificar imágenes del dataset MNIST.
Por lo que estamos trabajando con pesos.
 1. get_parameters: Envía los pesos actuales del modelo LOCAL.

 2. fit:
  2.1 Recibe los pesos globales del servidor.
    2.2 Entrena al modelo.
    2.3 Devuelve los pesos.

 3. evaluate: evaluamos el modelo con los datos de prueba.
    3.1 Recibe los pesos globales del servidor.
    3.2 Evalúa el modelo.
    3.3 Devuelve la pérdida y la precisión.

Referencia: https://flower.dev/docs/
"""

class FlowerClient(fl.client.NumPyClient): #Definir un cliente Flower que implementa los métodos necesarios para el entrenamiento y evaluación federados

    def get_parameters(self, config=None):
        return model.get_weights()

    def fit(self, parameters, config=None):
        model.set_weights(parameters) #Establecer los pesos del modelo recibido del servidor.
        model.fit(x_train_c, y_train_c, epochs=1, batch_size=32, verbose=0) #Una época es un ciclo completo a través del conjunto de datos. batch_size es el número de muestras que se procesan antes de actualizar los pesos del modelo. Verbose=0 significa que no se muestra salida durante el entrenamiento.
        return model.get_weights(), len(x_train_c), {} #Devolvemos los pesos y el número de muestras usadas.

    def evaluate(self, parameters, config=None):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=2)
        return loss, len(x_test), {"accuracy": acc}

"""
**Iniciar el cliente Flower**
Iniciamos el cliente Flower, conectándonos al servidor en la dirección "server:8080

Referencia: https://flower.dev/docs/
"""

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="server:8080", client=FlowerClient())
