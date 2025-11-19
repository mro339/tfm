import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# ----------------------
# Cargar datos MNIST (cada cliente usa una parte)
# ----------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:500] / 255.0
y_train = y_train[:500]
x_test = x_test[:100] / 255.0
y_test = y_test[:100]

# ----------------------
# Modelo
# ----------------------
def build_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ----------------------
# Cliente Flower
# ----------------------
class MnistClient(fl.client.NumPyClient):

    def __init__(self):
        self.model = build_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return float(loss), len(x_test), {"accuracy": float(acc)}

# ----------------------
# Ejecutar cliente
# ----------------------
print("📡 Conectando cliente al servidor federado...")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=MnistClient()
)
