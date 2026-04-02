FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir -e ".[dev]"
COPY mnemo /app/mnemo
EXPOSE 8000
CMD ["uvicorn", "mnemo.app.main:app", "--host", "0.0.0.0", "--port", "8000"]