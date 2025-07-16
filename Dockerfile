# Use a slim Python base image for smaller size and faster builds
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

# Set a working directory for the app
WORKDIR /code

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# (Recommended) Use /data for persistent storage (for PDFs, indexes, etc.)
# If you want, create these folders (commented out, since your code creates them as needed)
# RUN mkdir -p /data/textbooks /data/uploaded_pdfs

# Expose port 7860 (required for Hugging Face Spaces); also supports local 8000 if needed
EXPOSE 7860

# Set persistent storage path as environment variable (best practice for HF Spaces)
ENV HF_HOME=/data

# Start FastAPI app on port 7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]