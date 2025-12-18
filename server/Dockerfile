FROM python:3.10-slim
#Imagen base de Python 3.10 en su versión slim para reducir el tamaño de la imagen
WORKDIR /app
#Establece el directorio de trabajo dentro del contenedor en /app
#Añadimos matplotlib para posteriormente crear una gráfica.
RUN pip install --no-cache-dir flwr tensorflow==2.12
#Instala las dependencias necesarias: Flower (flwr) y TensorFlow versión 2.12 sin usar caché para reducir el tamaño de la imagen
COPY server.py .
#Copia el archivo server.py desde el host al directorio de trabajo del contenedor
CMD ["python", "server.py"]
#Define el comando por defecto para ejecutar el servidor cuando se inicie el contenedor
