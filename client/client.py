import os # Para manejar variables de entorno. Aquí se usa para obtener el ID del cliente desde una variable de entorno.
import flwr as fl
import tensorflow as tf
import numpy as np
import json
import datetime
import subprocess # Para ejecutar comandos de linux
import time
#import flex.data
#from flex.data import Dataset, FedDatasetConfig, FedDataDistribution


def configurar_red_adversa():
    """
    Usa 'tc' (Traffic Control) de Linux para simular una red mala.
    Requiere cap_add: NET_ADMIN en docker-compose.
    """
    latencia = os.environ.get("NET_LATENCY", "0ms")
    perdida = os.environ.get("NET_LOSS", "0%")
    ancho_banda = os.environ.get("NET_BANDWIDTH", "1000mbit")
    tipo_dispositivo = os.environ.get("PERFIL", "Desconocido")
    
    print(f"CONFIGURANDO RED ({tipo_dispositivo}):")
    print(f"   -> Latencia: {latencia} | Pérdida: {perdida} | Banda: {ancho_banda}")

    try:
        # 1. Limpiamos reglas anteriores (por si acaso)
        subprocess.run("tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
        
        # 2. Aplicamos las nuevas reglas (NetEm = Network Emulator)
        # rate: limita velocidad de descarga/subida
        # delay: añade ping
        # loss: tira paquetes aleatoriamente
        comando = f"tc qdisc add dev eth0 root netem delay {latencia} loss {perdida} rate {ancho_banda}"
        
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print("   ✅ Red configurada con éxito.")
        else:
            print(f"   ⚠️ Error configurando red (¿Falta NET_ADMIN?): {resultado.stderr}")
            
    except Exception as e:
        print(f"   ❌ Excepción configurando red: {e}")

# --- LLAMADA INICIAL ---
configurar_red_adversa()




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

#Obtenemos el ID de cada cliente y el número total de clientes, definidos en el sistema de orígenes. 
# Ver el docker-compose.yml o generate_compose.py.
client_id = int(os.environ.get("CLIENT_ID", "1"))
num_clients = int(os.environ.get("TOTAL_CLIENTS", "2"))

#Para identficarlo en los logs.
print(f" Cliente ID: {client_id} de {num_clients}")

def partition_data(x, y, client_id, num_clients, method="dirichlet", alpha=0.5, balance_quantity=True):
    """
    Divide los datos según los métodos descritos en el paper FLEX y el TFM.
    
    Métodos:
    - "iid": Reparto aleatorio uniforme.
    - "pathological": (Label Quantity Skew) Ordena por etiqueta y corta. Caso extremo.
    - "dirichlet": (Label Dirichlet Skew) Usa distribución Dirichlet para variar la 
                   probabilidad de cada clase en cada cliente.
                   alpha pequeño (0.1) = Muy Non-IID (desbalanceado).
                   alpha grande (100) = Casi IID (balanceado).
    """
    # IMPORTANTE: Fijar la semilla para que todos los clientes en Docker
    # calculen exactamente la misma matriz de distribución.
    np.random.seed(42) 
    
    if method == "iid":
        # Reparto aleatorio simple
        idxs = np.random.permutation(len(x))
        partition_idxs = np.array_split(idxs, num_clients)[client_id - 1]
        
    elif method == "pathological":
        # Ordenar por etiquetas y dividir en trozos. Es el método anterior que tenía.
        # Esto simula "Label Quantity Skew" donde cada cliente tiene pocas etiquetas
        idxs_sorted = np.argsort(y)
        partition_idxs = np.array_split(idxs_sorted, num_clients)[client_id - 1]
        
    elif method == "dirichlet":
        n_classes = len(np.unique(y))
        min_size = 0
        min_require_size = 10 
        N = len(x)
        idxs = np.arange(N)
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(n_classes):
                idx_k = idxs[y == k]
                np.random.shuffle(idx_k)
                
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # --- CAMBIO CLAVE AQUÍ ---
                if balance_quantity:
                    # El "freno" original: evita que uno se llene demasiado
                    proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                
                # Normalizamos siempre para que sume 1
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_batch_split = np.split(idx_k, proportions)
                for i in range(num_clients):
                    idx_batch[i] += idx_batch_split[i].tolist()
            
            min_size = min([len(idx_j) for idx_j in idx_batch])

        partition_idxs = np.array(idx_batch[client_id - 1])
        np.random.shuffle(partition_idxs)
        
    else:
        raise ValueError(f"Método '{method}' no reconocido.")

    return x[partition_idxs], y[partition_idxs]

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
# "pathological" -> Muy extremo (uno o pocos números por cliente, no se solapan). Es el caso más difícil.
# "dirichlet"    -> Realista (desbalanceado suave o fuerte según alpha)
# "iid"          -> Perfecto (irreal)

DISTRIBUTION_METHOD = "dirichlet" 
DIRICHLET_ALPHA = 0.1 # Cuanto más pequeño, más desbalanceado. 0.1 es muy desbalanceado, 1 es casi balanceado (IID).
DIRICHLET_BALANCE_QUANTITY = True # Si True, se asegura que ningún cliente tenga demasiados datos (freno para clientes con mucho más datos que otros).

# Aplicar partición a TRAIN
x_train_c, y_train_c = partition_data(x_train, y_train, client_id, num_clients, method=DISTRIBUTION_METHOD, alpha=DIRICHLET_ALPHA, balance_quantity=DIRICHLET_BALANCE_QUANTITY)

# Aplicar partición a TEST (Para evaluación realista en dominio local)
x_test_c, y_test_c = partition_data(x_test, y_test, client_id, num_clients, method=DISTRIBUTION_METHOD, alpha=DIRICHLET_ALPHA)

print(f"DATOS ASIGNADOS ({DISTRIBUTION_METHOD}):")
print(f"   -> Train: {len(x_train_c)} imgs. Etiquetas únicas: {np.unique(y_train_c)}")
print(f"   -> Test:  {len(x_test_c)} imgs. Etiquetas únicas: {np.unique(y_test_c)}")

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

Para garantizar la convergencia del algoritmo Federated Averaging, se asegura una inicialización común. 
En la ronda 1, el servidor selecciona los parámetros de un cliente arbitrario y los distribuye a toda la federación, 
asegurando que todos los clientes comiencen el descenso de gradiente desde la misma posición en el espacio de parámetros.

Esto ocurre porque si cada cliente comienza con una inicialización diferente, 
el proceso de agregación podría no converger a un modelo global óptimo, 
ya que los clientes podrían estar actualizando sus modelos en direcciones divergentes.

¡¡Esto ocurre dentro de la libreria de Flower, por lo que no es necesario implementarlo manualmente!!

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
        loss, acc = model.evaluate(x_test_c, y_test_c, verbose=2)
        ronda +=1
        #GLOBAL Current Round: El número de ronda actual, que se recibe del servidor a través del diccionario config. Si no se encuentra, se asigna "desconocida".
        mi_resultado = {
            "tiempo": str(datetime.datetime.now()), #a que hora se ha ejecutado. hora, minutos y segundos
            "loss": loss,
            "accuracy": acc,
            "data_size": len(x_test_c),
            "labels": str(np.unique(y_test_c)) # Guardamos qué números tenía este cliente
        }

        # Crear carpeta results si no existe (por seguridad)
        if not os.path.exists("/app/results"):
            os.makedirs("/app/results", exist_ok=True)
        
        # Guardamos en mi propio fichero usando mi ID
        archivo_propio = f"/app/results/client_{client_id}_metrics.json"
        
        with open(archivo_propio, "a") as f:
            f.write(json.dumps(mi_resultado) + "\n")
        

        print(f"📝 Cliente {client_id}: Resultado guardado (Acc: {acc:.4f})")


        
        return loss, len(x_test_c), {"accuracy": acc, "loss": loss} #Devolvemos la pérdida, el número de muestras usadas y un diccionario con métricas adicionales (en este caso, la precisión).

        
"""
**Iniciar el cliente Flower**
Iniciamos el cliente Flower, conectándonos al servidor en la dirección "server:8080

Referencia: https://flower.dev/docs/
"""

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="server:8080", client=FlowerClient())
