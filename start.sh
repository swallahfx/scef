#!/bin/bash

# SCEF Starter Script
# This script starts the SCEF web interface and services

echo "Starting SCEF - Strategy Composition and Evaluation Framework"
echo "------------------------------------------------------------"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 to use SCEF."
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python3 -m pip install -r requirements.txt

# Start the web interface
echo "Starting web interface..."
cd web
python3 -m uvicorn app:app --host 0.0.0.0 --port 8011 --reload &
WEB_PID=$!

echo "Web interface started on http://localhost:8011"
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "echo 'Stopping services...'; kill $WEB_PID; echo 'All services stopped.'; exit 0" INT

# Keep script running
while true; do
    sleep 1
done
