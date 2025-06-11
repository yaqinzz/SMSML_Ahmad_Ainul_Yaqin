#!/bin/bash

# Script to deploy the Lung Cancer Model stack
# This script assumes Docker and Docker Compose are installed

echo "Deploying Lung Cancer Model Stack..."

# Pull the latest model image from Docker Hub
echo "Pulling latest model image from Docker Hub..."
docker pull yaqin7/lung-cancer-model:latest

echo "Building the Docker Compose stack..."
docker-compose build
# Start the stack with Docker Compose
echo "Starting the stack with Docker Compose..."
docker-compose up -d

# Wait for services to be fully up
echo "Waiting for services to start up..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo "Deployment complete! The following endpoints are available:"
echo "- Model API: http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (login with yaqin77/admin)"

# Instructions to test the API
echo ""
echo "To test the API, run the following command:"
echo "curl -X POST http://localhost:8000/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"features\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}'"
echo ""
echo "Make sure to replace the feature values with actual values matching your model requirements!"
