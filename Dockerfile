# Use an official Python runtime as a parent image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory
WORKDIR /app

COPY . /app

RUN uv sync --frozen
RUN uv pip install "ctranslate2==4.5.0" 

# Expose the port FastAPI runs on
EXPOSE 8000
EXPOSE 8265

# Command to run the FastAPI app
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
