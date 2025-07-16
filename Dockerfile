FROM python:3.10-slim

# Install system dependencies for PyMuPDF, Tesseract OCR, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download required NLTK data at build time
RUN python -m nltk.downloader punkt stopwords

COPY . .

EXPOSE 7860

# Set Hugging Face and Transformers cache directories to a writable location
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]