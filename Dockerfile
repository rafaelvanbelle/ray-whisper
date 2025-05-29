# Use an official Python runtime as a parent image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory
WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen
#RUN uv pip install "ctranslate2==4.5.0" 

# Expose the port FastAPI runs on
EXPOSE 8000
EXPOSE 8265

# Make sure start.sh is executable
RUN chmod +x /app/start.sh

# Command to run the FastAPI app
#ENV RAY_SERVE_HTTP_HOST=0.0.0.0
#ENV RAY_SERVE_HTTP_PORT=8000

# Set the entrypoint to run the start.sh script
CMD ["/app/start.sh"]

