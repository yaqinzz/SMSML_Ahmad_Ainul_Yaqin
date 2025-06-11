import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
X_train_scaled = pd.read_csv('Lung_Cancer_preprocessing/X_train_scaled.csv')
X_test_scaled = pd.read_csv('Lung_Cancer_preprocessing/X_test_scaled.csv')
y_train = pd.read_csv('Lung_Cancer_preprocessing/y_train.csv')
y_test = pd.read_csv('Lung_Cancer_preprocessing/y_test.csv')

# Convert target variables to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Set up MLflow tracking
# dagshub.init(repo_owner='yaqinzz', repo_name='my-first-repo', mlflow=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Experiment Lung Cancer Model")
mlflow.sklearn.autolog()

# Train model and log metrics with MLflow
with mlflow.start_run():
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')  # Use macro for multiclass
    rec = recall_score(y_test, y_pred, average='macro')      # Use macro for multiclass
    f1 = f1_score(y_test, y_pred, average='macro')          # Use macro for multiclass
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Print metrics
    print("Model dilatih dan dicatat di MLflow.")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    
    # Save metrics as JSON and log as artifact
    metric_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    
    
    

