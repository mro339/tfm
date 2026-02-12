#Imagen base de Python 3.10 en su versión slim para reducir el tamaño de la imagen
FROM python:3.10-slim
#Establece el directorio de trabajo dentro del contenedor en /app
WORKDIR /app

#Instalar iproute2 para usar tc en el contenedor y simular condiciones de red
RUN apt-get update && apt-get install -y iproute2 && rm -rf /var/lib/apt/lists/*

#Instala las dependencias necesarias: Flower (flwr) y TensorFlow versión 2.12 sin usar caché para reducir el tamaño de la imagen
RUN pip install --no-cache-dir flwr tensorflow==2.12
#Copia el archivo server.py desde el host al directorio de trabajo del contenedor
COPY server.py .

#Define el comando por defecto para ejecutar el servidor cuando se inicie el contenedor
CMD ["python", "server.py"]
