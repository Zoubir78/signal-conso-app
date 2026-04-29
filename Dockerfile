# Dockerfile (racine)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV API_URL=http://api:8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . .

# Dependency installation
RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

EXPOSE 8501

CMD ["streamlit", "run", "src/app/dashboard/dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]