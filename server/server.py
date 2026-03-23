import flwr as fl
import os
from typing import List, Tuple
from flwr.common import Metrics
import json

"""
**Servidor Flower** 
1. Precisión del modelo.
 1.1 Recibimos la lista con pesos y métricas de cada cliente. En función al número del conjunto de datos.
 1.2 Obtenemos el número total de clientes. 
2. Establecemos una estrategia.
3. Inicializamos el servidor que escuche.

"""
def evaluate_config(server_round: int):
    #Envía configuracion a los clientes para la evaluación
    return {"server_round": server_round}

#Calculamos como está funcionando el modelo, haciendo una media con todos los clientes y su conjunto de datos.
CURRENT_ROUND = 0
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global CURRENT_ROUND
    CURRENT_ROUND += 1
    
    #Extraemos el número de ejemplos (imágenes) de cada cliente
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    
    #Calculamos las medias ponderadas de TODAS las métricas
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]

    global_accuracy = sum(accuracies) / total_examples
    global_precision = sum(precisions) / total_examples
    global_recall = sum(recalls) / total_examples
    global_f1_score = sum(f1_scores) / total_examples


    #Obtenemos los clientes que particiapn y activos.
    clientes_participantes = [m.get("client_id", "Desconocido") for _, m in metrics]
    num_clientes_activos = len(clientes_participantes)

    '''
        Prueba----borrar luego
        "num_clientes_activos": num_clientes_activos,
        "clientes_participantes": clientes_participantes
    '''

    resultado_ronda = {
        "ronda": CURRENT_ROUND,
        "num_clientes_activos": len(clientes_participantes),
        "num_clientes_activos": num_clientes_activos,
        "clientes_participantes": clientes_participantes,
        "metricas_globales": {
            "accuracy": global_accuracy,
            "precision": global_precision,
            "recall": global_recall,
            "f1_score": global_f1_score
        },
        # Guardamos el detalle individual por si lo necesitamos
        "detalle_clientes": [
            {
                "client_id": m.get("client_id", "Desconocido"),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "muestras": num_examples
            } for num_examples, m in metrics
        ]
    }

    with open("/app/results/global_results.json", "a") as f:
        f.write(json.dumps(resultado_ronda) + "\n")
    
    print(f"Ronda: {CURRENT_ROUND} | Clientes: {clientes_participantes} | Acc Global: {global_accuracy:.4f} | F1 Global: {global_f1_score:.4f}")
    
    return {"accuracy": global_accuracy, "f1_score": global_f1_score}


#Aqui modificamos el código, para obtener el número total de clientes.
total_clients = int(os.environ.get("TOTAL_CLIENTS", "2"))
min_clients = int(total_clients*0.6)
if min_clients<1: min_clients=1

strategy = fl.server.strategy.FedAvg( #Define la estrategia de agregación federada
    fraction_fit=1.0, #Porcentaje de clientes que participan en cada ronda de entrenamiento 1=100% (TODOS) Reducimos cuando tenemos muchos clientes
    fraction_evaluate=1.0, #Porcentaje de clientes que participan en cada ronda de evaluación 1=100% (TODOS) la diferecncia con fit es que evalua el modelo despues de entrenar y fit es entrenar
    min_fit_clients=min_clients, #Número mínimo de clientes que deben participar en el entrenamiento por ronda
    min_evaluate_clients=min_clients, #Número mínimo de clientes que deben participar en la evaluación por ronda
    min_available_clients=min_clients, #Número mínimo de clientes que deben estar disponibles para que el servidor inicie una ronda // Antes estaba todos, ahora solo el mínimo
    evaluate_metrics_aggregation_fn=weighted_average,
    on_evaluate_config_fn=evaluate_config #Pasa la ronda a los clientes.
  )


if __name__ == "__main__": # Punto de entrada del servidor. Si el archivo se ejecuta directamente, se inicia el servidor federado
    # Limpiar fichero anterior al arrancar
    #if os.path.exists("/app/results/global_results.json"):
    #    os.remove("/app/results/global_results.json")

    print(f"Servidor iniciado con estrategia FedAvg. Esperando a {total_clients} clientes...")
    fl.server.start_server( 
        server_address="0.0.0.0:8080", #Escucha en todas las interfaces de red en el puerto 8080
        config=fl.server.ServerConfig(num_rounds=10), #Número de rondas de entrenamiento federado
        strategy=strategy
    )


