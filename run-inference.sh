#!/bin/bash
# run-inference.sh - Script to run the inference application with continuous operation and security

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to validate the environment before starting the application
validate_environment() {
    # Check if required files/directories exist
    if [ ! -f "logging_config.json" ]; then
        echo "Error: logging_config.json not found."
        exit 1
    fi
    
    if [ ! -f "src/inference/inference_service.py" ]; then
        echo "Error: Inference service file not found."
        exit 1
    fi
    
    # Additional checks can be added here as needed
    
    echo "Environment validation passed."
}

# Function to start the inference application with restart capability
start_inference() {
    echo "Starting inference application..."
    source .venv/bin/activate
    
    # Run the application with custom logging configuration
    uvicorn src.inference.inference_service:app \
        --host 0.0.0.0 \
        --port 8001 \
        --reload \
        --log-config logging_config.json
}

# Main loop to restart the application if it crashes
while true; do
    validate_environment
    if start_inference; then
        echo "Inference application stopped normally."
        break
    else
        EXIT_CODE=$?
        echo "Inference application crashed with exit code $EXIT_CODE. Restarting in 5 seconds..."
        sleep 5
    fi
done
