# **Simulación de Aprendizaje Federado en un entorno realista.**

**Trabajo de fin de máster**

**Autor**: Miguel Ángel Rodríguez Ortega

---

## **Descripción del Proyecto**

Este proyecto implementa un simulador avanzado de Aprendizaje Federado (Federated Learning) utilizando **Flower (flwr)** y **TensorFlow**. Su objetivo principal es evaluar el rendimiento de modelos bajo condiciones reales y hostiles de redes distribuidas.

El simulador incorpora:
- **Heterogeneidad de Red y Hardware:** Simulación de dispositivos IoT, móviles y conexiones Wi-Fi con limitaciones reales de CPU, ancho de banda, latencia y pérdida de paquetes mediante la herramienta `tc qdisc (NetEm)` de Linux.
- **Heterogeneidad de Datos (Non-IID):** Particionado de datos desbalanceado simulando casos de uso reales mediante distribuciones de Dirichlet y "Pathological skew".
- **Dinámica de Clientes:** Simulación de conexiones tardías (*Late Joining*) y caídas abruptas de clientes (*Dropouts*) empleando técnicas de Ingeniería del Caos con Pumba.

---

## **Requisitos Previos**

Para ejecutar este simulador, necesitas tener instalado en tu máquina host:
- **Docker** y **Docker Compose** (Versión V2 recomendada).
- **Python 3.10** o superior (únicamente para ejecutar el script generador).

---

## **Guía de comandos básicos de ejecución**
Todos los comandos deben ejecutarse desde la terminal, en la raíz del proyecto.


1. **Generar la infraestructura (docker-compose.yml):**
   Este script crea dinámicamente el archivo de Docker Compose en base al número de clientes y perfiles que hayas configurado.

    ```bash
    pyhton docker-compose.yml
    ```
2. **Levantar y ejecutar la simulación:**
Este comando construye las imágenes Docker (si no están creadas) e inicia el servidor y todos los clientes en paralelo.
    ```bash
    docker compose up --build
    ```
3. **Detener la simulación y limpiar:**
    ```bash
    docker compose down
    ```
---

## **Mapa de configuración de cada parámetro**
El proyecto está diseñado de forma modular. Aquí tienes la guía exacta de dónde ir para modificar los parámetros de tus experimentos.

1. **Infraestructura y Red (En `generate_compose.py`)**
    - **Número total de clientes**: Variable `NUM_CLIENTS`.

    - **Perfiles de Hardware/Red**: Diccionario `PERFILES`. Aquí puedes editar la CPU (`cpu`), latencia (`latencia`), pérdida de paquetes (`loss`) y ancho de banda (`banda`) para cada tipo de dispositivo (IoT, Móvil, WiFi, Servidor).

    - **Late Joining (Conexión tardía)**: En la generación de cada cliente, modifica la variable de entorno `START_DELAY` (ej. `random.choice([0, 30, 60])` para que tarden entre 0 y 60 segundos en unirse).

    - **Dropouts (Ingeniería del Caos)**: Al final del script, en la configuración del servicio pumba. Puedes cambiar el intervalo de eliminación en la línea `command: --random --interval 2m kill --signal SIGTERM "re2:^fl-client"`.

2. **Parámetros del Aprendizaje Federado (En `server.py`)**
    Controla cómo el servidor coordina el entrenamiento:

    - **Número de Rondas (Epochs globales)**: Modifica el parámetro `num_rounds=10` dentro de `fl.server.ServerConfig()`.

    - **Tolerancia a fallos**: Modifica la variable `min_clients = int(total_clients * 0.5)` para decidir qué porcentaje de clientes vivos es necesario para que el servidor inicie o continúe una ronda sin quedarse bloqueado.

3. **Datos y Modelo Local (En `client.py`)**
    Controla el comportamiento interno de cada dispositivo:

    - **Método de distribución de datos**: Variable `DISTRIBUTION_METHOD` (Opciones: `"dirichlet"`, `"pathological"`, `"iid"`).

    - **Grado de desbalanceo (Non-IID)**: Variable `DIRICHLET_ALPHA`. Un valor de `0.1` es altamente desbalanceado (difícil); un valor de `1.0` o superior es más homogéneo (fácil).

    - **Épocas locales**: Modifica `epochs=1` dentro de `model.fit()` en el método `fit` del cliente.

---

## **Resultados y Logs**
La evaluación del modelo se guarda automáticamente en formato JSON estructurado, listo para ser analizado con Pandas o Jupyter Notebooks.

- Todos los resultados se almacenan en la carpeta `/results` generada automáticamente.

- `global_results.json`: Contiene la precisión, pérdida, recall y F1-Score global calculado por el servidor ronda a ronda, además del número de clientes que sobrevivieron en esa iteración.

- `client_[ID]_metrics.json`: Contiene las métricas locales y el desempeño individual de cada nodo frente a su propio conjunto de datos (Test set).

---