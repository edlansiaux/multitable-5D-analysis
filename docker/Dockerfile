# Dockerfile pour MT5D Analysis API
FROM python:3.9-slim

# Maintien des métadonnées
LABEL maintainer="Edouard Lansiaux <edouard1.lansiaux@chu-lille.fr>"
LABEL description="5D Multi-Table Analysis Framework API"

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY setup.py .
COPY mt5d/__init__.py mt5d/__init__.py

# Installation des dépendances Python
# Note: Pour la prod, on fixerait les versions dans requirements.txt
# Ici on utilise setup.py pour installer les libs requises + FastAPI
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir torch dgl -f https://data.dgl.ai/wheels/repo.html

# Copie du code source complet
COPY . .

# Création des dossiers de sortie
RUN mkdir -p outputs configs

# Exposition du port API
EXPOSE 8000

# Commande de démarrage
CMD ["python", "mt5d/api/main.py"]
