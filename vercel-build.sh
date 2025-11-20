#!/bin/bash

# Vercel build script
set -e  # Exit on any error

echo "Starting Vercel build process..."

# Install frontend dependencies and build Tailwind CSS
echo "Installing Node.js dependencies..."
npm install

echo "Building Tailwind CSS..."
npm run build:tailwind

echo "Build process completed successfully!"