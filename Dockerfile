# Base Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx 
    && rm -rf /var/lib/apt/lists/*  

# Set working directory in container
WORKDIR /app

# Copy requirements file first (for better Docker cache utilization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt  # --no-cache-dir reduces image size

# Copy all project files from host to container
COPY . .

# Default command to run when container starts
CMD ["python", "inference.py"]  # Runs inference script by default
