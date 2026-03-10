FROM python:3.7-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for numpy/scipy/sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

CMD ["python", "venv_info.py"]