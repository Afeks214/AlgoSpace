# Start from a clean, standard Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy ONLY the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# --- CRITICAL STEP ---
# Upgrade pip and install all libraries from the requirements file.
# The --no-cache-dir flag ensures a clean install.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of the project source code
COPY . .

# Set the default command for when the container runs
CMD ["python3", "src/main.py"]