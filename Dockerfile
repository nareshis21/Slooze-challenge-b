# Use a more robust Python runtime as a parent image (Bookworm instead of Slim)
FROM python:3.10-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    DOCLING_MODELS_CACHE=/home/user/.cache/docling/models

# Create a non-root user
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Install critical system dependencies for Docling, OpenCV, and OCR
# libgl1-mesa-glx and libglib2.0-0 are for OpenCV
# libsm6, libxext6, libxrender1 are for UI-less PDF processing
# tesseract-ocr is for fallback OCR layers
# libmagic1 is for file type detection
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and set ownership
COPY --chown=user:user . .

# Ensure cache directories exist and are writable by the user
RUN mkdir -p .llama_cache $HF_HOME $DOCLING_MODELS_CACHE && \
    chown -R user:user $HOME

# Switch to the non-root user
USER user

# Hugging Face Spaces expect port 7860
EXPOSE 7860

# Run Streamlit with the correct port and address
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
