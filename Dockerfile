FROM python:3.13-slim

WORKDIR /app

# libgomp1 is required at runtime by LightGBM (it uses OpenMP for
# multi-threaded training/inference) - a common gotcha on slim base
# images that don't include it by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit exposes a built-in health endpoint - checked here without
# needing to install curl/wget separately, since Python is already
# available in the image.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]