import requests
import pandas as pd
import numpy as np
from io import StringIO

# Create a dummy ECG CSV
def create_dummy_csv():
    # Generate synthetic ECG data (just random noise for testing connectivity)
    # In a real scenario, we'd use real ECG data, but for API testing, 
    # we just need to ensure the pipeline runs without crashing.
    # The feature extraction might return zeros or weird values, but the model should still predict.
    ecg_data = np.random.rand(21000) - 0.5  # 30 seconds at 700 Hz
    df = pd.DataFrame({'ECG': ecg_data})
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_models():
    base_url = "http://localhost:8000"
    csv_content = create_dummy_csv()
    
    models = [
        "random_forest",
        "svc",
        "cnn",
        "decision_tree",
        "ensemble",
        "logistic_regression",
        "naive_bayes"
    ]
    
    print(f"Testing API at {base_url}...")
    
    # Check health first
    try:
        health = requests.get(f"{base_url}/health")
        print("Health Check:", health.json())
    except Exception as e:
        print(f"Failed to connect to backend: {e}")
        return

    for model in models:
        print(f"\nTesting model: {model}")
        files = {
            'file': ('test.csv', csv_content, 'text/csv')
        }
        data = {
            'model_type': model,
            'sampling_rate': '700'
        }
        
        try:
            response = requests.post(f"{base_url}/predict", files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Prediction: {result['prediction']['prediction_label']}")
            else:
                print(f"❌ Failed! Status: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_models()
