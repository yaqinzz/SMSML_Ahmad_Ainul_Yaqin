import requests
import random
import time
import pandas as pd

# Load a sample from the test dataset to use for predictions
try:
    X_test = pd.read_csv('Lung_Cancer_preprocessing/X_test_scaled.csv')
    print(f"Loaded {len(X_test)} test samples")
except Exception as e:
    print(f"Error loading test data: {e}")
    # Create some dummy data if we can't load the real test data
    X_test = pd.DataFrame([
        [0.1, 0.2, 0.3, 0.4] * 5 + [0.5, 0.6, 0.7]  # 23 features
    ])

def make_prediction(features, should_succeed=True):
    try:
        # Ensure features is a list of numbers
        features_list = [float(f) for f in features]
        
        # If we want this to fail, mess up the features
        if not should_succeed:
            if random.random() < 0.5:
                # Send wrong number of features
                features_list = features_list[:10]
                print("Sending intentionally wrong number of features")
            else:
                # Send invalid features
                features_list = ["invalid"] + features_list[1:]
                print("Sending intentionally invalid features")
        
        # Make API call
        print(f"Sending features: {features_list[:5]}... (total: {len(features_list)})")
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features_list},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Prediction: {response.json()}")
            return True
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def get_metrics():
    try:
        response = requests.get("http://localhost:8000/metrics")
        print(f"Metrics status: {response.status_code}")
        return True
    except Exception as e:
        print(f"Failed to get metrics: {e}")
        return False

# Generate some sample traffic
print("Generating sample API traffic...")
for i in range(30):  # Increased from 20 to 30
    # Randomly select a sample from the test dataset
    sample_idx = random.randint(0, len(X_test) - 1)
    sample = X_test.iloc[sample_idx].values.tolist()
    
    # Determine if this should be a successful or failing request
    # Make ~25% of requests fail
    should_succeed = random.random() > 0.25
    
    # Make prediction
    success = make_prediction(sample, should_succeed)
    
    # Get metrics occasionally
    if random.random() > 0.7:
        get_metrics()
    
    # Wait a bit
    time.sleep(random.uniform(0.5, 2.0))

print("Done generating traffic!")
