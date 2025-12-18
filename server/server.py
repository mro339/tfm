import flwr as fl
import os
from typing import List, Tuple
from flwr.common import Metrics

"""
**Servidor Flower** 
1. Precisión del modelo.
 1.1 Recibimos la lista con pesos y métricas de cada cliente. En función al número del conjunto de datos.
 1.2 Obtenemos el número total de clientes. 
2. Establecemos una estrategia.
3. Inicializamos el servidor que escuche.

"""

#Calculamos como está funcionando el modelo, haciendo una media con todos los clientes y su conjunto de datos.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

#Aqui modificamos el código, para obtener el número total de clientes.
total_clients = int(os.environ.get("TOTAL_CLIENTS", "2"))
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
    print("Servidor federado iniciado...")

    fl.server.start_server( 
        server_address="0.0.0.0:8080", #Escucha en todas las interfaces de red en el puerto 8080
        config=fl.server.ServerConfig(num_rounds=3), 
        strategy=strategy
    )


