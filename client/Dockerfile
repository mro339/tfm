FROM python:3.10-slim
#Imagen base de Python 3.10 en su versión slim para reducir el tamaño de la imagen
WORKDIR /app
#Establece el directorio de trabajo dentro del contenedor en /app
RUN pip install --no-cache-dir flwr tensorflow==2.12
#Instala las dependencias necesarias: Flower (flwr) y TensorFlow versión 2.12 sin usar caché para reducir el tamaño de la imagen
COPY client.py .
#Copia el archivo client.py desde el host al directorio de trabajo del contenedor
CMD ["python", "client.py"]
#Define el comando por defecto para ejecutar el cliente cuando se inicie el contenedor

#Este Dockerfile configura un entorno para ejecutar un cliente de aprendizaje federado utilizando Flower y TensorFlow. En resumen, establece una imagen base de Python, instala las dependencias necesarias, copia el archivo del cliente y define el comando para ejecutarlo.