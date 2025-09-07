# Pawikan Sentinel

Pawikan Sentinel is a wildlife monitoring system for detecting and tracking marine turtles (pawikan) using computer vision.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Python package manager and project runner

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

## Running the Applications

### Web Application

To run the web application:
```bash
./run-web.sh
```

The web application will be available at http://localhost:8000

### Inference Application

To run the inference application:
```bash
./run-inference.sh
```

The inference application will be available at http://localhost:8001

## Project Structure

- `src/` - Source code
  - `src/api/` - API routers
  - `src/core/` - Core application files
  - `src/inference/` - Inference-related code
  - `src/web/` - Web templates and static files
- `docs/` - Documentation
- `deployments/` - Deployment scripts and configs
- `tests/` - Test files