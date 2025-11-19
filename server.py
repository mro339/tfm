import flwr as fl

# Estrategia FedAvg simple
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,      # Usar todos los clientes conectados para entrenar
    fraction_evaluate=1.0, # Todos evalúan
    min_fit_clients=2,     # Necesitamos 2 clientes
    min_available_clients=2,
    min_evaluate_clients=2,
)

# Iniciar servidor
if __name__ == "__main__":
    print("🚀 Iniciando servidor federado en 0.0.0.0:8080 ...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
