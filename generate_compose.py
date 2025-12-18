
"""
**Generador de docker-compose.yml**
He tenido que crear este script, para poder generar un archivo docker-compose.yml con tantos clientes como quiera.
En docker-compose.yml no se pueden usar bucles. Y como cada cliente y servidor tiene que estar definido, he creado este script, dado que todos los clientes son iguales únicammente cambiando el ID.

El archivo docker.compose.yml se encarga de generar y levantar los contenedores Dockers para el entorno.
Por lo tanto, tenemos que defenir quienes actúan, que sería el servidor y los clientes.
Realmente al ejecturarse, se lee desde abajo hacia arriba, levantando primero la red, como se conectan los contenedores a la red.

1. RED flnet: federated learning network:
La conexión que va a tener es de tipo bridge (puente). Es la de por defecto.
la red Bridge funciona como un punete virtual, conecta los distintos contenedores entre sí y con el host, permitiendo una comuncación fluida y privada.
No se necesita configuración adicional, ya que Docker gestiona automáticamente la asignación de direcciones IP y el enrutamiento dentro de la red bridge.
Otras opciones son: host (usa la tarjeta red de mi ordenador), none (no hay red), overlay (avanzado) permite conexiones con ordenadores físicos diferente) y Macvlan (asigna direcciones MAC a los contenedores).

2. Servidor:
 2.1 build: ./server. Indicamos la ruta donde se encuentra el Dockerfile
 2.2 Puerto: 8080:8080. 

 8080              :               8080
Entrada           Conecta         Espera escuchar

 2.3 Variables de entorno para luego poder usar el número de clientes en otro script.

3. Clientes:
 3.1 build: ./client. Indicamos la ruta donde se encuentra el Dockerfile
 3.2 Depencia en el servidor, no empieza hasta que no haya arrancado el servidor
 3.3 Variables de entorno para posteriormente usar el ID del cliente y el número total de clientes.
 
CUESTIONES PARA PREGUNTAR AL PROFESOR:
Cosas que se pueden añadir a la configuración de la red:
Latencia, pérdida de paquetes, limitación de ancho de banda, aislamiento de red, configuración de DNS personalizada, etc.
Latencia: Al ser bridge, no hay latencia... (overlay??) o añadir tiempos de espera en la comunicación
Ancho de banda: Usan todos mis CPU.
Fallos de red: Pérdida de paquetes, caídas de conexión.
Heterogeneidad: Todos usan la misma CPU y RAM.

--- Propuestas:
Limitar recursoss:

limits:
  cpus: '0.50'
  memory: 512M
  
Añadir fallos de red:
  deploy:
    resources:
      reservations:
        devices:
          - driver: bridge
            capabilities: [ latency, loss, bandwidth ]
            options:
              latency: 100ms
              loss: 10%
              bandwidth: 1mbit+


pumba:
  image: gaiaadm/pumba
  command: netem --duration 5m delay --time 1000 jitter 50 client1


https://docs.docker.com/compose/compose-file/compose-file-v3/#resources
https://docs.docker.com/config/containers/resource_constraints/

Referencias:
https://www.youtube.com/watch?v=LkcTF8BYVTU&t=190s
https://docs.docker.com/compose/compose-file/
https://www.youtube.com/watch?v=-l7YocEQtA0&t=40s
https://docs.docker.com/get-started/docker_cheatsheet.pdf --Como iniciar dockers.

-Levantar los contenedores:
docker compose up --build 
-Para bajar los contenedores:
docker compose down
-Para eliminar los ocntendeores
docker rm -f "nombres de los contenedores"
-Para ver los logs
docker compose logs -f server
"""
import sys

# Configuración
# Puedes cambiar esto o pasarlo por argumento
NUM_CLIENTS = 2

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

print(f"Archivo docker-compose.yml generado con {NUM_CLIENTS} clientes.")