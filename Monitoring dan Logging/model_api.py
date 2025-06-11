import time
import os
import psutil
import threading
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Define metrics
# Using a common prefix 'api_' to ensure consistent naming in Prometheus
PREDICTION_RESULT = Gauge('api_prediction_result', 'Hasil prediksi terakhir (0, 1 dan 2).')
PREDICTION_CONFIDENCE = Gauge('api_prediction_confidence', 'Tingkat kepercayaan dari prediksi terakhir.')
TOTAL_REQUESTS = Counter('api_total_requests', 'Jumlah total request yang dikirim ke model.')
SUCCESSFUL_REQUESTS = Counter('api_successful_requests', 'Jumlah request yang berhasil diproses model.')
FAILED_REQUESTS = Counter('api_failed_requests', 'Jumlah request yang gagal diproses model.')
RESPONSE_LATENCY = Histogram('api_response_latency_seconds', 'Waktu respons dari model server dalam detik.')

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM

# Add two new metrics
ENDPOINT_REQUEST_TIME = Histogram('endpoint_request_time_seconds', 'Waktu respons per endpoint API.', 
                                ['endpoint', 'method', 'status_code'])
MEMORY_PER_REQUEST = Gauge('memory_per_request_bytes', 'Penggunaan memori untuk setiap request prediksi.')

# Initialize FastAPI app
app = FastAPI(title="Lung Cancer Prediction Service")

# Define request data model
class PredictionRequest(BaseModel):
    features: list

# Configuration for model server
import os
import requests
import json
import numpy as np

# Determine if we're running in Docker or locally
def is_running_in_docker():
    """Check if we're running inside a Docker container"""
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return any('docker' in line for line in f)
    except:
        return False

# If we're running in Docker, use the service name, otherwise use localhost
if is_running_in_docker():
    MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model-server:8080")
else:
    MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8080")
    
MODEL_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"
print(f"Using model server URL: {MODEL_SERVER_URL}")

# Function to make prediction using the model server
def predict_with_model_server(features):
    headers = {"Content-Type": "application/json"}

    global feature_names
    if feature_names and len(feature_names) == len(features):
        columns = feature_names
    else:
        columns = [f"feature_{i}" for i in range(len(features))]

    # First, try to get probabilities
    payload_proba = {
        "dataframe_split": {
            "columns": columns,
            "data": [features],
            "index": [0]
        },
        "params": {"method": "predict_proba"}
    }

    try:
        # Try to get probabilities first
        response = requests.post(MODEL_ENDPOINT, headers=headers, data=json.dumps(payload_proba))
        response.raise_for_status()
        result = response.json()
        print(f"Model server response for predict_proba: {result}")

        # Handle different response formats
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("No predictions returned from model server")
        
        # Get the first prediction result
        prediction_result = predictions[0]
        
        # Check if we got probabilities (list/array) or just a single prediction
        if isinstance(prediction_result, (list, tuple, np.ndarray)):
            # We got probabilities
            probabilities = list(prediction_result)
            prediction = int(np.argmax(probabilities))  # Get class with highest probability
            confidence = max(probabilities)
            return prediction, probabilities, confidence
        else:
            # We got a single prediction, fall back to regular predict
            print("predict_proba returned single value, falling back to regular predict")
            raise ValueError("predict_proba not supported, falling back to predict")
            
    except Exception as e:
        print(f"Error with predict_proba: {e}, falling back to regular predict")
        
        # Fallback to regular predict
        payload_predict = {
            "dataframe_split": {
                "columns": columns,
                "data": [features],
                "index": [0]
            }
        }
        
        try:
            response = requests.post(MODEL_ENDPOINT, headers=headers, data=json.dumps(payload_predict))
            response.raise_for_status()
            result = response.json()
            print(f"Model server response for predict: {result}")
            
            predictions = result.get("predictions", [])
            if not predictions:
                raise ValueError("No predictions returned from model server")
            
            prediction = int(predictions[0])
            
            # Since we don't have probabilities, create a mock probability distribution
            # This is not ideal, but provides a fallback
            num_classes = 3  # Assuming 3 classes based on your lung cancer model
            probabilities = [0.0] * num_classes
            probabilities[prediction] = 1.0  # Set predicted class to 100%
            confidence = 1.0
            
            return prediction, probabilities, confidence
            
        except Exception as fallback_error:
            print(f"Error with fallback predict: {fallback_error}")
            raise fallback_error

print(f"Model API configured to use model server at: {MODEL_SERVER_URL}")

# Feature names from the training data
feature_names = None
try:
    # Try to load feature names, supports both relative and absolute paths
    feature_paths = [
        'Lung_Cancer_preprocessing/feature_names.txt',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Lung_Cancer_preprocessing/feature_names.txt')
    ]
    
    for path in feature_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(feature_names)} feature names from {path}")
            break
    
    if feature_names is None:
        print("Could not find feature_names.txt in the expected locations")
except Exception as e:
    print(f"Error loading feature names: {e}")
    # Handle the case when feature names are not available

@app.get("/")
async def root():
    return {"message": "Lung Cancer Prediction API is running"}

@app.get("/health")
async def health():
    try:
        # Check if model server is accessible
        response = requests.get(f"{MODEL_SERVER_URL}/ping")
        if response.status_code == 200:
            return {"status": "healthy", "model_server": "connected"}
        else:
            raise HTTPException(status_code=500, detail=f"Model server returned status {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model server connection error: {str(e)}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    TOTAL_REQUESTS.inc()
    start_time = time.time()

    try:
        expected_features = len(feature_names) if feature_names else 23
        if len(request.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature count. Expected {expected_features} features, got {len(request.features)}."
            )

        try:
            # Use the improved prediction function
            prediction, probabilities, confidence = predict_with_model_server(request.features)
            # Log ke Prometheus
            PREDICTION_RESULT.set(prediction)
            PREDICTION_CONFIDENCE.set(confidence)
            SUCCESSFUL_REQUESTS.inc()

            elapsed_time = time.time() - start_time
            RESPONSE_LATENCY.observe(elapsed_time)

            return {
                "prediction": prediction,
                "prediction_probabilities": probabilities,
                "confidence": confidence,
                "latency_seconds": elapsed_time
            }
        except Exception as e:
            FAILED_REQUESTS.inc()
            raise HTTPException(status_code=500, detail=f"Error from model server: {str(e)}")

    except HTTPException:
        FAILED_REQUESTS.inc()
        elapsed_time = time.time() - start_time
        RESPONSE_LATENCY.observe(elapsed_time)
        raise
    except Exception as e:
        FAILED_REQUESTS.inc()
        elapsed_time = time.time() - start_time
        RESPONSE_LATENCY.observe(elapsed_time)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    # Update system metrics before returning
    cpu_percent = psutil.cpu_percent()
    CPU_USAGE.set(cpu_percent)
    
    memory = psutil.virtual_memory()
    RAM_USAGE.set(memory.percent)
    
    # Explicitly check that metrics are properly registered
    print(f"Available metrics: TOTAL_REQUESTS={TOTAL_REQUESTS._value.get()}, SUCCESSFUL_REQUESTS={SUCCESSFUL_REQUESTS._value.get()}, FAILED_REQUESTS={FAILED_REQUESTS._value.get()}")
    
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Record memory before request
    mem_before = psutil.Process().memory_info().rss
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record endpoint metrics
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    # Record the request time in the histogram
    ENDPOINT_REQUEST_TIME.labels(endpoint=endpoint, method=method, status_code=status_code).observe(process_time)
    
    # Calculate memory used for this request (only for predict endpoint)
    if endpoint == "/predict":
        mem_after = psutil.Process().memory_info().rss
        memory_used = mem_after - mem_before
        MEMORY_PER_REQUEST.set(memory_used)
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Function to update system metrics
def update_system_metrics():
    """Update system metrics (CPU and RAM usage) every second"""
    while True:
        # Update CPU usage
        cpu_percent = psutil.cpu_percent()
        CPU_USAGE.set(cpu_percent)
          # Update RAM usage
        memory = psutil.virtual_memory()
        RAM_USAGE.set(memory.percent)
        
        # Sleep for 1 second before the next update
        time.sleep(1)

if __name__ == "__main__":
    # Start system metrics monitoring in a background thread
    metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
    metrics_thread.start()
    
    # Start the FastAPI application
    import uvicorn
    print("Starting server with system metrics monitoring...")
    uvicorn.run(app, host="0.0.0.0", port=8000)