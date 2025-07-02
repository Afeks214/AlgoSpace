# AlgoSpace Docker Quick Start Guide

## ✅ Docker Environment Successfully Set Up!

Your Docker environment is now configured and ready for AlgoSpace development.

### 🚀 Quick Start

#### On Linux/Mac:
```bash
chmod +x start-dev.sh
./start-dev.sh
```

#### On Windows:
```cmd
start-dev.bat
```

### 📦 What Was Done

1. **Docker Mode**: Verified Docker is running in Linux container mode
2. **Image Built**: Created lightweight `algospace-env` image (351MB)
3. **Environment Tested**: Python 3.9 environment is operational
4. **Scripts Created**: Development launch scripts for both platforms

### 🛠️ Available Docker Images

- `algospace-env`: Lightweight image with minimal dependencies
- Use `Dockerfile` for full environment with torch/scikit-learn
- Use `Dockerfile.light` for faster builds during development

### 🔧 Common Commands

Inside the container:
```bash
# Install full dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific component
python src/main.py

# Install Jupyter for notebooks
pip install jupyter
jupyter notebook --ip=0.0.0.0 --allow-root
```

### ⚠️ Known Issues

1. **Tests require torch**: Install with `pip install torch` inside container
2. **Limited disk space**: The full torch installation is large (~2GB)
3. **Windows mode switching**: May need manual Docker Desktop intervention

### 💡 Tips

- Use volume mounts to edit code on host while running in container
- Ports 8000 and 8888 are exposed for web services
- Container removes itself on exit (`--rm` flag)
- All changes to code are persistent (volume mounted)

### 🔄 Rebuilding

If you need to rebuild the image:
```bash
# Clean build
docker build --no-cache -t algospace-env .

# Remove old images
docker system prune -af
```

---
Environment is ready for development! 🎉