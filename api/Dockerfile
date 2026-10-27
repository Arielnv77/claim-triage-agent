# Lightweight base image with Python 3.11
FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# Install the dependencies first to take advantage of the Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app ./app

# Application tests, runnable with "make test" inside the container
COPY tests ./tests

# The API listens on port 8000
EXPOSE 8000

# Start uvicorn listening on all interfaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
