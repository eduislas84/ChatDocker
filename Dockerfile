# Imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia solo el archivo de requerimientos e instálalo
COPY CapitanPeru2/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia únicamente los archivos esenciales del proyecto
COPY CapitanPeru2/telegram_csv_bot.py .
COPY CapitanPeru2/.env .

# Comando para ejecutar el bot
CMD ["python", "telegram_csv_bot.py"]
