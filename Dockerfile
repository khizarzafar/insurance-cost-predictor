# ============================================================
# Dockerfile — Insurance Charges Predictor
# ============================================================
# A Dockerfile is a recipe. Each line is a step.
# Docker executes these steps top-to-bottom to BUILD an image.
# An image is like a snapshot of a fully configured computer.
# A container is a running instance of that image.
#
# Build the image:   docker build -t insurance-predictor .
# Run a container:   docker run -p 7860:7860 -p 8000:8000 insurance-predictor
# ============================================================

# ── Step 1: Choose a base image ──────────────────────────────
# Think of this as "start with a freshly installed operating system"
# python:3.11-slim = Python 3.11 on a minimal Linux (Debian)
# We use 3.11 (not 3.13) because it has the most stable package ecosystem
# 'slim' = no extra tools pre-installed → smaller image size
FROM python:3.11-slim

# ── Step 2: Set the working directory ────────────────────────
# All following commands will run from this folder inside the container
# It's like doing: cd /app  (and creating /app if it doesn't exist)
WORKDIR /app

# ── Step 3: Install system dependencies ──────────────────────
# Some Python packages need system-level C libraries to compile.
# We install them here before installing Python packages.
# --no-install-recommends = don't install extra tools we don't need
# rm -rf /var/lib/apt/lists/* = clean up apt cache to keep image size small
RUN apt-get update && apt-get install -y \
    build-essential \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ── Step 4: Copy and install Python dependencies ─────────────
# We copy requirements.txt BEFORE copying our code.
# Why? Docker caches each step. If our code changes but requirements.txt doesn't,
# Docker reuses the cached pip install step — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 5: Copy our project files into the container ────────
# The dot (.) means "copy everything in the current folder"
# into the WORKDIR (/app) of the container
COPY . .

# ── Step 6: Make the startup script executable ───────────────
# In Linux, scripts need explicit permission to run.
# 'chmod +x' gives execution permission to start.sh
RUN chmod +x start.sh

# ── Step 7: Expose ports ─────────────────────────────────────
# This documents which ports the container uses.
# It does NOT automatically publish them — you still need -p when running.
# 8000 = FastAPI backend
# 7860 = Streamlit frontend (Hugging Face Spaces requires port 7860)
EXPOSE 8000
EXPOSE 7860

# ── Step 8: Set the startup command ──────────────────────────
# This is what runs when someone starts the container.
# It calls our start.sh which launches both services.
CMD ["./start.sh"]
