import flwr as fl
import os
from typing import List, Tuple
from flwr.common import Metrics


#Definir la función de agregación de métricas
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Esta función recibe una lista de (num_ejemplos, métricas) de cada cliente
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Calcula la media ponderada (Weighted Average)
    # Da más importancia a la nota de los clientes que tienen más datos
    return {"accuracy": sum(accuracies) / sum(examples)}

#Aqui modificamos el código, para obtener el número total de clientes.
total_clients = int(os.environ.get("TOTAL_CLIENTS", "2")) #Número total de clientes (debe coincidir con el número de instancias de cliente que se ejecutan)
min_clients = int(total_clients*0.6)

if min_clients<1: min_clients=1

strategy = fl.server.strategy.FedAvg( #Define la estrategia de agregación federada
    fraction_fit=1.0, #Porcentaje de clientes que participan en cada ronda de entrenamiento 1=100% (TODOS)
    fraction_evaluate=1.0, #Porcentaje de clientes que participan en cada ronda de evaluación 1=100% (TODOS) la diferecncia con fit es que evalua el modelo despues de entrenar y fit es entrenar
    min_fit_clients=min_clients, #Número mínimo de clientes que deben participar en el entrenamiento por ronda
    min_evaluate_clients=min_clients, #Número mínimo de clientes que deben participar en la evaluación por ronda
    min_available_clients=total_clients, #Número mínimo de clientes que deben estar disponibles para que el servidor inicie una ronda
    evaluate_metrics_aggregation_fn=weighted_average,
  )


if __name__ == "__main__": # Punto de entrada del servidor. Si el archivo se ejecuta directamente, se inicia el servidor federado
    print("🚀 Servidor federado iniciado...")

    fl.server.start_server( #inicia el servidor federado con la configuración especificada
        server_address="0.0.0.0:8080", #Escucha en todas las interfaces de red en el puerto 8080
        config=fl.server.ServerConfig(num_rounds=3), #Configura el servidor para ejecutar 3 rondas de entrenamiento federado
        strategy=strategy
    )

#Flower se ha encargado de gestionar la comunicación entre el servidor y los clientes, así como de coordinar el proceso de entrenamiento federado.
#Levanta un servidor gRCP (un servidor de comunicación) que escucha las conexiones entrantes de los clientes federados. en el puerto 8080
#Espera a que los clientes se conecten y participen en el proceso de entrenamiento federado.
#Por cada ronda de entrenamiento federado, el servidor coordina la selección de clientes, la distribución de los parámetros del modelo, la recopilación de las actualizaciones del modelo y la agregación de estas actualizaciones para mejorar el modelo global.
#Envia instrucciones a los clientes para que realicen segun strategy
#Gestiona la sincronización entre los clientes y el servidor para asegurar que todos los participantes estén alineados en cada ronda de entrenamiento.
#Después de completar el número especificado de rondas, el servidor puede guardar el modelo final o realizar evaluaciones adicionales según sea necesario.
