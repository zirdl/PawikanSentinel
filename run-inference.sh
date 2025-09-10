#!/bin/bash
# run-inference.sh - Script to run the inference application with proper environment variables and logging

# Stop any existing containers
echo "Stopping existing inference containers..."
docker stop $(docker ps -q --filter ancestor=roboflow/roboflow-inference-server-cpu) 2>/dev/null || echo "No existing containers to stop"

# Create the matplotlib config directory if it doesn't exist
mkdir -p /app/.matplotlib

# Set the MPLCONFIGDIR environment variable to fix the Matplotlib issue
export MPLCONFIGDIR=/app/.matplotlib

# Pull the latest image to ensure we have the most up-to-date version
echo "Pulling latest inference server image..."
docker pull roboflow/roboflow-inference-server-cpu:latest

# Run the Docker container
echo "Starting inference server..."
CONTAINER_ID=$(docker run -d -p 9001:9001 --name pawikan-inference roboflow/roboflow-inference-server-cpu:latest)

if [ $? -eq 0 ]; then
    echo "Inference server started successfully with container ID: $CONTAINER_ID"
    echo "Server is available at http://localhost:9001"
    echo "To view logs, run: docker logs $CONTAINER_ID"
    echo "To stop the server, run: docker stop $CONTAINER_ID"
else
    echo "Failed to start inference server"
    exit 1
fi