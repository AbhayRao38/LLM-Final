FROM python:3.10-slim

# Install system dependencies for PyMuPDF, Tesseract, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for your app
WORKDIR /code

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all files to the container
COPY . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Set persistent storage path as environment variable (for HF Spaces best practice)
ENV HF_HOME=/data

# Start FastAPI app on port 7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]