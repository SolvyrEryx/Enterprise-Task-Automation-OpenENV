FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY app.py ./
COPY inference.py ./
COPY test_demo.py ./
COPY openenv.yaml ./
COPY README.md ./

# Install dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir gradio openai

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Health check — verifies the environment package loads correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from src import EnterpriseEnv; env = EnterpriseEnv(); env.reset(); print('OK')" || exit 1

# Launch the Gradio web interface (required for HF Spaces)
CMD ["python", "app.py"]
