#!/bin/bash
# Startup script for data pipeline

echo "Starting AlgoSpace Data Pipeline..."

# Set environment
export PYTHONPATH=/app:$PYTHONPATH

# Check config file
CONFIG_FILE=${1:-"config/data_pipeline.yaml"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"

# Start with monitoring
exec python scripts/deploy_data_pipeline.py "$CONFIG_FILE"