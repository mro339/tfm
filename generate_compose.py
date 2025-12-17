##Con este archivo lo que haremos será generar un docker-compose.yml dinámicamente, para poder especificar el número de clientes que queremos levantar.
##Por lo tanto simplemente generaremos un bucle que añada los servicios de cliente según el número especificado.
#Dado a que realmente lo único que cambia entre los clientes es la variable de entorno CLIENT_ID.


import sys

# Configuración
# Puedes cambiar esto o pasarlo por argumento
NUM_CLIENTS = 5 

# Plantilla del encabezado y el servidor (que siempre es igual)
yaml_content = f"""
name: federated-learning

services:
  server:
    build: ./server
    container_name: fl-server
    ports:
      - "8080:8080"
    networks:
      - flnet
    environment:
      - TOTAL_CLIENTS={NUM_CLIENTS} 

"""

# Bucle para generar los clientes
for i in range(1, NUM_CLIENTS + 1):
    yaml_content += f"""
  client{i}:
    build: ./client
    container_name: fl-client{i}
    environment:
      - CLIENT_ID={i}
      - TOTAL_CLIENTS={NUM_CLIENTS}
    depends_on:
      - server
    networks:
      - flnet
"""

# Añadir la red al final
yaml_content += """
networks:
  flnet:
    driver: bridge
"""

# Guardar el archivo
with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Archivo docker-compose.yml generado exitosamente con {NUM_CLIENTS} clientes.")