#!/bin/bash
# build-css.sh - Script to build Tailwind CSS

echo "Building Tailwind CSS..."

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed."
    exit 1
fi

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build Tailwind CSS
npx tailwindcss -i ./src/web/static/css/input.css -o ./src/web/static/css/tailwind.css

echo "Tailwind CSS build completed!"