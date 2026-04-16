# DejaVuGuard — Multi-stage Dockerfile
# Stage 1: Build frontend
# Stage 2: Python runtime with static files

# ─── Stage 1: Frontend build ───
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ─── Stage 2: Python runtime ───
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy backend code
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Copy built frontend static files
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create data directory for SQLite
RUN mkdir -p /data

# Environment defaults
ENV DATABASE_PATH=/data/dejavuguard.db
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
