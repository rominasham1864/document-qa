# Local development image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install PyTorch independently
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]