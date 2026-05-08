FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding model and build cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8765

CMD ["uvicorn", "web_ui:app", "--host", "0.0.0.0", "--port", "8765"]
