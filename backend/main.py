from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk
from io import StringIO
from typing import Dict, List
import traceback

app = FastAPI(title="ECG Stress Detection API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/svc.pkl', 'rb') as f:
        svc_model = pickle.load(f)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = None
    svc_model = None


def extract_hrv_features(ecg_signal: np.ndarray, sampling_rate: int = 700) -> Dict:
    """
    Extract time-domain HRV features using NeuroKit2
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling rate of the ECG signal
    
    Returns:
        Dictionary containing HRV time-domain features matching model requirements
    """
    try:
        # Clean the ECG signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        
        # Calculate HRV time-domain features
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
        
        # Define the exact 19 features that models were trained on
        required_features = [
            "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
            "HRV_CVNN", "HRV_CVSD", "HRV_MedianNN", "HRV_MadNN",
            "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN",
            "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN",
            "HRV_MaxNN", "HRV_HTI", "HRV_TINN"
        ]
        
        # Extract only the required features
        features = {}
        for feature_name in required_features:
            if feature_name in hrv_time.columns:
                features[feature_name] = float(hrv_time[feature_name].iloc[0])
            else:
                # If feature is missing, set to 0 or NaN
                print(f"Warning: Feature {feature_name} not found in extracted features")
                features[feature_name] = 0.0
        
        return features
    
    except Exception as e:
        raise ValueError(f"Error extracting HRV features: {str(e)}")


def predict_stress(features: Dict, model_type: str = "random_forest") -> Dict:
    """
    Predict stress class using the specified model
    
    Args:
        features: Dictionary of HRV features
        model_type: "random_forest" or "svc"
    
    Returns:
        Dictionary with prediction and probability
    """
    model = rf_model if model_type == "random_forest" else svc_model
    
    if model is None:
        raise ValueError(f"Model {model_type} not loaded")
    
    # Get expected feature names from the model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        
        # Create feature dict with only expected features
        aligned_features = {}
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name in features:
                aligned_features[feature_name] = features[feature_name]
            else:
                missing_features.append(feature_name)
                aligned_features[feature_name] = 0.0  # Default value for missing features
        
        if missing_features:
            print(f"Warning: Missing features filled with 0: {missing_features}")
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([aligned_features])[expected_features]
    else:
        # If model doesn't have feature_names_in_, use all features
        feature_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        try:
            probability = model.predict_proba(feature_df)[0]
            prob_dict = {
                "no_stress": float(probability[0]),
                "stress": float(probability[1])
            }
        except Exception as e:
            print(f"Could not get probabilities: {e}")
            prob_dict = None
    elif hasattr(model, 'decision_function'):
        # For SVC without probability, use decision function
        try:
            decision = model.decision_function(feature_df)[0]
            # Convert decision function to probability-like scores
            # Using sigmoid function
            import math
            prob_stress = 1 / (1 + math.exp(-decision))
            prob_no_stress = 1 - prob_stress
            prob_dict = {
                "no_stress": float(prob_no_stress),
                "stress": float(prob_stress)
            }
        except Exception as e:
            print(f"Could not get decision function: {e}")
            prob_dict = None
    else:
        prob_dict = None
    
    return {
        "prediction": int(prediction),
        "prediction_label": "Stress" if prediction == 1 else "No Stress",
        "probabilities": prob_dict,
        "probability_method": "predict_proba" if hasattr(model, 'predict_proba') else ("decision_function" if hasattr(model, 'decision_function') else "none")
    }


@app.get("/")
def read_root():
    return {
        "message": "ECG Stress Detection API",
        "endpoints": {
            "/predict": "POST - Upload CSV file for stress prediction",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
def health_check():
    model_features = {}
    
    # Try to get feature names from models
    if rf_model is not None and hasattr(rf_model, 'feature_names_in_'):
        model_features['random_forest'] = rf_model.feature_names_in_.tolist()
    
    if svc_model is not None and hasattr(svc_model, 'feature_names_in_'):
        model_features['svc'] = svc_model.feature_names_in_.tolist()
    
    return {
        "status": "healthy",
        "models_loaded": {
            "random_forest": rf_model is not None,
            "svc": svc_model is not None
        },
        "expected_features": model_features
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("random_forest"),
    sampling_rate: int = Form(700)
):
    """
    Process ECG CSV file and predict stress level
    
    Args:
        file: CSV file containing ECG signal data
        model_type: Model to use ("random_forest" or "svc")
        sampling_rate: Sampling rate of ECG signal (default: 700 Hz)
    
    Returns:
        JSON with features, prediction, and probabilities
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Extract ECG signal from the ECG column
        if 'ECG' in df.columns:
            ecg_signal = df['ECG'].values
        elif 'ecg' in df.columns:
            ecg_signal = df['ecg'].values
        else:
            # Fallback to first column if ECG column not found
            ecg_signal = df.iloc[:, 0].values
        
        # Extract HRV features
        hrv_features = extract_hrv_features(ecg_signal, sampling_rate)
        
        # Predict stress
        prediction_result = predict_stress(hrv_features, model_type)
        
        # Prepare response
        response = {
            "success": True,
            "features": hrv_features,
            "prediction": prediction_result,
            "model_used": model_type,
            "signal_info": {
                "length": len(ecg_signal),
                "sampling_rate": sampling_rate,
                "duration_seconds": len(ecg_signal) / sampling_rate
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/predict-both")
async def predict_both_models(
    file: UploadFile = File(...),
    sampling_rate: int = 700
):
    """
    Process ECG CSV file and predict using both models
    
    Returns predictions from both Random Forest and SVC models
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Extract ECG signal from the ECG column
        if 'ECG' in df.columns:
            ecg_signal = df['ECG'].values
        elif 'ecg' in df.columns:
            ecg_signal = df['ecg'].values
        else:
            # Fallback to first column if ECG column not found
            ecg_signal = df.iloc[:, 0].values
        
        # Extract HRV features
        hrv_features = extract_hrv_features(ecg_signal, sampling_rate)
        
        # Predict with both models
        rf_prediction = predict_stress(hrv_features, "random_forest")
        svc_prediction = predict_stress(hrv_features, "svc")
        
        # Prepare response
        response = {
            "success": True,
            "features": hrv_features,
            "predictions": {
                "random_forest": rf_prediction,
                "svc": svc_prediction
            },
            "signal_info": {
                "length": len(ecg_signal),
                "sampling_rate": sampling_rate,
                "duration_seconds": len(ecg_signal) / sampling_rate
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)