# Lung Cancer Prediction Model - Monitoring and Logging

This folder contains components for deploying and monitoring a Lung Cancer prediction model. The monitoring system uses Prometheus for metrics collection and Grafana for visualization, allowing complete observability of the model in production.

## Project Structure

- `Lung_Cancer_preprocessing/`: Preprocessed data for model evaluation and testing
- `docker-compose.yml`: Docker Compose configuration for running the full monitoring stack
- `model_api.py`: FastAPI application for serving predictions with integrated metrics
- `prometheus.yml`: Prometheus configuration for monitoring
- `generate_traffic.py`: Script to generate test traffic and evaluate the API
- `deploy.sh`: Deployment script for setting up the monitoring stack
- `requirements-api.txt`: Dependencies for the FastAPI application
- `Dockerfile.api`: Docker configuration for building the API service

## System Overview

This monitoring system provides complete observability for the deployed lung cancer prediction model:

1. FastAPI application serves predictions and exposes metrics
2. Prometheus collects and stores metrics from both API and model server
3. Grafana visualizes the metrics with customized dashboards
4. Traffic generator script helps test the system under various conditions

## Getting Started

### Prerequisites

- Python 3.10
- Docker and Docker Compose
- Git

### Installation

1. Navigate to the Monitoring and Logging directory:

   ```bash
   cd "Monitoring dan Logging"
   ```

2. Deploy the stack using the deployment script:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

The deployment script will:

- Pull the latest model image from Docker Hub
- Start all services with Docker Compose
- Provide information on available endpoints

### Testing the API

You can test the API by sending a POST request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [feature_values_here]}'
```

Replace `feature_values_here` with the appropriate feature values for the model.

Alternatively, you can use the included traffic generator script to automatically send test requests:

```bash
python generate_traffic.py
```

This script loads test data from the preprocessing folder and sends both successful and intentionally failed requests to test all aspects of the API, including error handling.

### Monitoring

- **Prometheus**: Available at http://localhost:9090
- **Grafana**: Available at http://localhost:3000 (login with user: yaqin77, password: admin)

#### API Metrics

The API exposes the following custom metrics through Prometheus:

- `api_prediction_result`: Gauge that shows the latest prediction result (0, 1 dan 2)
- `api_prediction_confidence`: Gauge that shows the confidence level of the latest prediction
- `api_total_requests_total`: Counter for the total number of requests sent to the model
- `api_successful_requests_total`: Counter for successfully processed requests
- `api_failed_requests_total`: Counter for failed requests
- `api_response_latency_seconds`: Histogram of response times from the model server
- `endpoint_request_time_seconds`: Histogram of API response times by endpoint, HTTP method and status code
- `memory_per_request_bytes`: Gauge tracking memory usage per prediction request

#### System Metrics

- `system_cpu_usage`: CPU usage percentage
- `system_ram_usage`: RAM usage percentage

You can query these metrics in Prometheus or visualize them using the provided Grafana dashboard.

## Development

### Making API Changes

The API is built with FastAPI and serves predictions by communicating with the model server:

1. Update `model_api.py` with your changes
2. Rebuild the API container:
   ```bash
   docker-compose build app
   docker-compose up -d app
   ```

### Adding Custom Metrics

To add new metrics to the monitoring system:

1. Define new Prometheus metrics in `model_api.py`
2. Update the metrics collection logic in the API
3. Create new panels in the Grafana dashboard to visualize the metrics

## Components

The monitoring stack includes:

- **Model Server**: Serves the ML model using MLflow's serving capabilities
- **API Server**: FastAPI application that communicates with the model server
- **Prometheus**: Collects metrics from both servers
- **Grafana**: Visualizes the collected metrics

### System Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│             │      │             │      │             │
│   Client    ├─────►│  FastAPI    ├─────►│  MLflow     │
│             │      │  Server     │      │  Model      │
│             │      │             │      │  Server     │
└─────────────┘      └──────┬──────┘      └─────────────┘
                            │
                            │
                     ┌──────▼──────┐
                     │             │
                     │ Prometheus  │
                     │             │
                     └──────┬──────┘
                            │
                            │
                     ┌──────▼──────┐
                     │             │
                     │  Grafana    │
                     │             │
                     └─────────────┘
```

### Container Dependencies

- The API server depends on the model server
- Prometheus depends on the API server
- Grafana depends on Prometheus

This ensures that services are started in the correct order.

## Monitoring Components

### FastAPI with Integrated Metrics

The `model_api.py` file implements a FastAPI application with Prometheus metrics integration:

- Uses the Prometheus client library to expose metrics
- Tracks prediction results, confidence levels, and request latency
- Monitors system-level metrics like CPU and RAM usage

### Prometheus Configuration

The `prometheus.yml` file configures data collection:

- Scrapes metrics from the API server at regular intervals
- Collects data from the MLflow model server
- Supports local development targets for testing

### Grafana Dashboard

The pre-configured Grafana dashboard displays:

- Model prediction performance metrics
- System resource utilization
- API health and performance indicators
- Request and error rates

## Troubleshooting

### Common Issues

1. **Model Server Connection Error**

   - Check if the model server is running: `docker-compose ps`
   - Verify the MODEL_SERVER_URL environment variable in docker-compose.yml

2. **Prometheus Data Source Error in Grafana**

   - Ensure that the Prometheus data source in Grafana uses `http://prometheus:9090` as the URL (not localhost)
   - Restart the Grafana container if changes don't take effect

3. **Missing Metrics in Prometheus**
   - Check API logs: `docker-compose logs app`
   - Verify that metrics are being exposed at http://localhost:8000/metrics
   - Ensure Prometheus is scraping the correct targets: http://localhost:9090/targets

### Logs

To view logs from any component:

```bash
docker-compose logs [service]
```

Where `[service]` is one of: app, model-server, prometheus, or grafana.

## License

MIT License

## Contributors

Project developed as part of the "Membangun Sistem Machine Learning" course by LASKAR AI.

## Acknowledgements

- MLflow for model tracking and serving
- FastAPI for API development
- Prometheus and Grafana for monitoring
- Docker for containerization
