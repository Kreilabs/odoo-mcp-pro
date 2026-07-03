FROM python:3.11-slim

WORKDIR /app

# Copiar los archivos del repositorio
COPY . /app

# Instalar las dependencias y el propio paquete en modo ejecutable
RUN pip install --no-cache-dir . mcp[cli] fastapi uvicorn

# Exponer el puerto que exige Cloud Run
EXPOSE 8080

# Comando de inicio: Corre el servidor MCP usando transporte SSE en el puerto 8080
CMD ["python", "-m", "mcp_server_odoo", "--transport", "streamable-http", "--port", "8080", "--host", "0.0.0.0"]