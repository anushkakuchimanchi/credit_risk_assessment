# save_model.py - Save your trained model and preprocessors
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Assuming you have your trained model and data
# Replace 'xgb_smote' with your actual trained model variable name

def save_model_artifacts(model, scaler=None, feature_names=None, target_encoder=None):
    """
    Save all necessary model artifacts for Streamlit app
    
    Parameters:
    - model: Your trained XGBoost model
    - scaler: Fitted StandardScaler (if used)
    - feature_names: List of feature names used in training
    - target_encoder: LabelEncoder for target variable (if used)
    """
    
    # Save the main model
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved successfully!")
    
    # Save scaler if provided
    if scaler is not None:
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Scaler saved successfully!")
    
    # Save feature names
    if feature_names is not None:
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        print("✅ Feature names saved successfully!")
    
    # Save target encoder if provided
    if target_encoder is not None:
        with open('models/target_encoder.pkl', 'wb') as f:
            pickle.dump(target_encoder, f)
        print("✅ Target encoder saved successfully!")

# Example usage - Run this after training your model
if __name__ == "__main__":
    # Replace these with your actual variables
    
    # Your trained model (replace with actual variable name)
    # model = xgb_smote  # Your trained XGBoost model
    
    # Your fitted scaler (if you used one)
    # scaler = your_scaler  # Your fitted StandardScaler
    
    # Feature names from your training data
    # feature_names = list(x_train.columns)  # Your feature column names
    
    # Uncomment and run these lines after replacing with your actual variables:
    
    # save_model_artifacts(
    #     model=model,
    #     scaler=scaler,
    #     feature_names=feature_names
    # )
    
    print("Model artifacts ready for Streamlit deployment!")
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run credit_app.py")
    print("2. Update the load_model() function in credit_app.py")
    print("3. Test with sample predictions")

# Create a requirements.txt file for deployment
