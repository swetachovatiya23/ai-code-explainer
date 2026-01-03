FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Set Streamlit config directory
ENV STREAMLIT_HOME=/home/user/.streamlit
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /home/user/app

# Create necessary directories
RUN mkdir -p /home/user/.streamlit /home/user/.cache

# Copy requirements first for better caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--server.maxUploadSize=200", "--server.headless=true"]