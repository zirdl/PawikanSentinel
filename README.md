# Pawikan Sentinel

Pawikan Sentinel is a wildlife monitoring system for detecting and tracking marine turtles (pawikan) using computer vision.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Python package manager and project runner
- Node.js and npm (for building Tailwind CSS)

## Setup

1. Install Python dependencies:
   ```bash
   uv sync
   ```

2. Install Node.js dependencies for Tailwind CSS:
   ```bash
   npm install
   ```

## Running the Applications

### Web Application

To run the web application:
```bash
./run-web.sh
```

The web application will be available at http://localhost:8000

This script will automatically build the Tailwind CSS before starting the application.

### Inference Application

To run the inference application:
```bash
./run-inference.sh
```

The inference application will be available at http://localhost:8001

## Building Tailwind CSS

The web application uses Tailwind CSS with custom colors. The CSS is built automatically when running the web application, but you can also build it manually:

```bash
./build-css.sh
```

For development, you can watch for changes and automatically rebuild the CSS:

```bash
./watch-css.sh
```

## Project Structure

- `src/` - Source code
  - `src/api/` - API routers
  - `src/core/` - Core application files
  - `src/inference/` - Inference-related code
  - `src/web/` - Web templates and static files
- `docs/` - Documentation
- `deployments/` - Deployment scripts and configs
- `tests/` - Test files