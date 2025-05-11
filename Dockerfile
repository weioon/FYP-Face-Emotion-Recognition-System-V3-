# ---- Stage 2: Build Backend ----
FROM python:3.10-slim
WORKDIR /app
# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Install system dependencies if needed (e.g., for OpenCV)
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Copy backend requirements first for caching
COPY ./Backend/requirements.txt /app/Backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/Backend/requirements.txt
# Copy the rest of the backend code
COPY ./Backend /app/Backend
# Copy the built frontend files from the builder stage to the 'static' directory
# Copy any other necessary files/models from the root needed by the backend
COPY ./yolov8n.pt /app/yolov8n.pt
# COPY ./evaluation_model.py /app/evaluation_model.py

# Expose the port the app runs on (uvicorn default is 8000)
EXPOSE 8000

# Set the working directory inside Backend for running the app
WORKDIR /app/Backend

# Command to run the application (use the actual internal port)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]