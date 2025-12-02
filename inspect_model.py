import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

def inspect_ensemble():
    try:
        with open('backend/models/ensemble.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model type: {type(model)}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names: {model.feature_names_in_}")
        elif hasattr(model, 'get_booster'):
            print(f"Booster feature names: {model.get_booster().feature_names}")
        else:
            print("Could not determine feature names directly.")
            
        # Try to predict with dummy data to see if we can trigger a more informative error or success
        # Try with 5 features
        try:
            dummy_5 = pd.DataFrame(np.zeros((1, 5)), columns=[f'f{i}' for i in range(5)])
            model.predict(dummy_5)
            print("Successfully predicted with 5 dummy features.")
        except Exception as e:
            print(f"Failed with 5 dummy features: {e}")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_ensemble()
