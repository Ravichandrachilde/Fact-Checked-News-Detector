FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/tmp/huggingface_cache
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
RUN mkdir -p /tmp/huggingface_cache/hub && chmod -R 777 /tmp/huggingface_cache

RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio
COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
