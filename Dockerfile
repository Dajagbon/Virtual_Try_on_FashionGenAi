FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy code
COPY . /app

# Python deps (pin versions for reproducibility)
RUN pip3 install --no-cache-dir runpod requests pillow numpy torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122

# Expose handler start
CMD ["python3", "rp_handler.py"]
