FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg git && \
    pip install --upgrade pip && pip install -r requirements.txt

COPY app /app/app
COPY ui /app/ui
COPY run.sh /app/run.sh

CMD ["bash", "run.sh"]