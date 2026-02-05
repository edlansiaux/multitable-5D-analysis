FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Installer Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copier le code
COPY . .

# Installer en mode développement
RUN pip install -e .

# Exposer le port pour Jupyter/API
EXPOSE 8888 8000

# Commande par défaut
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
