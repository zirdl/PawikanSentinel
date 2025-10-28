#!/bin/bash
# This script starts the Roboflow inference server using the local Docker image.

echo "Starting Roboflow inference server..."

docker run -d --name roboflow-inference -p 9001:9001 -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -v inference-cache:/tmp/cache roboflow/roboflow-inference-server-cpu:latest
