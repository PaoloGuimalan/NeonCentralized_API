# --- Stage 1: Build Stage ---
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies into wheels to save space in final image
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: Final Runtime Stage ---
FROM python:3.13-slim

# Set recommended production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8003

WORKDIR /app

# Install only essential runtime system libraries
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install them
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the rest of your application code
COPY . .
COPY production.env .env

# Security: Create and use a non-root user
RUN useradd -m django_user && chown -R django_user /app
USER django_user

# Expose the production port
EXPOSE 8004

# Run with Gunicorn instead of manage.py runserver
# Workers formula: (2 * cores) + 1
CMD ["gunicorn", "--bind", "0.0.0.0:8004", "--workers", "3", "--timeout", "120", "neon.wsgi:application"]