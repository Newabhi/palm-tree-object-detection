# Use the official NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Install Python and other necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install gdown for downloading model
RUN pip install gdown

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the model directory
RUN mkdir -p /app/model

# Download the model
RUN python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=1BUQYSwNvkHX5WmVfBN-3L9wgHuLxpe4L', '/app/model/model.pth', quiet=False)"

# Install pip requirements
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Expose the port that the FastAPI server will run on
EXPOSE 8000

# run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", 8000]
