FROM python:3.9-slim

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libgtk-3-dev \
    libwebkit2gtk-4.0-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia solo primero los archivos necesarios para instalar requirements (para aprovechar caché)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instala recursos de NLTK y spaCy
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_sm

# Ahora sí copiamos todo el código
COPY . .

# Expone el puerto que Streamlit usará
EXPOSE 10000

# Comando final para lanzar la app
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0", "--server.headless=true"]
