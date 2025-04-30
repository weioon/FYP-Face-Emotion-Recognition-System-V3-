# ---- Stage 1: Build Frontend ----
    FROM node:18-alpine as builder
    WORKDIR /app/frontend
    # Copy only package files first for caching
    COPY frontend/package.json frontend/package-lock.json* ./
    RUN npm install
    # Copy the rest of the frontend source code
    COPY frontend/ ./
    # Set the API URL build-time argument (adjust if you prefixed API routes)
    ARG REACT_APP_API_URL=/api
    ENV REACT_APP_API_URL=${REACT_APP_API_URL}
    # Build the frontend
    RUN npm run build
    
    # ---- Stage 2: Build Backend ----
    FROM python:3.10-slim
    WORKDIR /app
    # Set environment variables for Python
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1
    # Install system dependencies if needed (e.g., for OpenCV)
    # RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
    # Copy backend requirements first for caching
    COPY Backend/requirements.txt ./
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt
    # Copy the rest of the backend code
    COPY Backend/ ./Backend/
    # Copy the built frontend files from the builder stage to the 'static' directory
    COPY --from=builder /app/frontend/build ./Backend/static
    
    # Expose the port the app runs on (uvicorn default is 8000)
    # DigitalOcean App Platform uses the PORT env var, which uvicorn respects via $PORT
    # EXPOSE 8000
    
    # Set the working directory inside Backend for running the app
    WORKDIR /app/Backend
    
    # Command to run the application (DO App Platform uses $PORT)
    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]