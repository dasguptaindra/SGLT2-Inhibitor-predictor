# =====================================================
# External Molecule Predictor (PRODUCTION READY)
# =====================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

# ---------- RDKit & Mordred ----------
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred.error import Error

# ---------- SHAP ----------
import shap

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURE_PATH = "model_features.json"

# =====================================================
# LOAD MODEL & FEATURES
# =====================================================
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        model_features = json.load(f)
    print(f"\n✅ Loaded model with {len(model_features)} features")
except FileNotFoundError as e:
    print(f"\n❌ Error loading files: {e}")
    print("Ensure 'gradient_boosting_model.joblib' and 'model_features.json' are in the directory.")
    exit()

# =====================================================
# DESCRIPTOR NAME MAPPING
# =====================================================
DESCRIPTOR_NAME_MAP = {
    "MINaaN": "MINaaN",
    "MAXaaN": "MAXaaN",
    "BCUTs-1h": "BCUTs-1h",
    "nHBAcc_Lipinski": "nHBAcc",
}

# =====================================================
# DESCRIPTOR CALCULATION
# =====================================================
def calculate_descriptors_from_smiles(smiles: str) -> pd.DataFrame:
    """Calculate Mordred descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Add hydrogens for better descriptor calculation
    mol = Chem.AddHs(mol)
    
    # Calculate all Mordred descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    desc_series = calc(mol)
    
    # Convert to dictionary with proper error handling
    desc_dict = {}
    for name, val in desc_series.items():
        col_name = str(name)
        if isinstance(val, Error):
            desc_dict[col_name] = np.nan
        else:
            try:
                desc_dict[col_name] = float(val)
            except (ValueError, TypeError):
                desc_dict[col_name] = np.nan
    
    return pd.DataFrame([desc_dict])

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(desc_df, required_features):
    """Extract required features from Mordred descriptors."""
    X = pd.DataFrame(0.0, index=[0], columns=required_features)
    available_columns = [str(c) for c in desc_df.columns]
    
    for feat in required_features:
        # Use mapped name if available, otherwise use original
        mordred_name = DESCRIPTOR_NAME_MAP.get(feat, feat)
        
        if mordred_name in available_columns:
            val = desc_df.at[0, mordred_name]
            if pd.notna(val) and val not in [np.inf, -np.inf]:
                X.at[0, feat] = float(val)
            else:
                X.at[0, feat] = 0.0  # Default for NaN/inf values
        else:
            # Try direct match with original name
            if feat in available_columns:
                val = desc_df.at[0, feat]
                if pd.notna(val) and val not in [np.inf, -np.inf]:
                    X.at[0, feat] = float(val)
                else:
                    X.at[0, feat] = 0.0
            else:
                X.at[0, feat] = 0.0  # Default for missing features
    
    return X

# =====================================================
# FEATURE SUMMARY
# =====================================================
def print_feature_summary(X, desc_df, model_features):
    """Print a summary of feature values."""
    available_columns = [str(c) for c in desc_df.columns]
    
    print("\n" + "="*50)
    print("FEATURE VALUE SUMMARY")
    print("="*50)
    
    for feat in model_features:
        mordred_name = DESCRIPTOR_NAME_MAP.get(feat, feat)
        value = X.at[0, feat]
        
        if mordred_name in available_columns:
            original_val = desc_df.at[0, mordred_name]
            status = "✅ Valid" if pd.notna(original_val) and original_val not in [np.inf, -np.inf] else "⚠️  NaN/Inf"
            print(f"{feat:20} ({mordred_name:20}): {value:10.6f} {status}")
        else:
            print(f"{feat:20} ({mordred_name:20}): {value:10.6f} ⚠️  Not in Mordred")
    
    non_zero_count = (X != 0).sum(axis=1).iloc[0]
    print(f"\n📊 Features with non-zero values: {non_zero_count}/{len(model_features)}")

# =====================================================
# PREDICTION PIPELINE
# =====================================================
def predict_smiles(smiles: str):
    """Run the complete prediction pipeline."""
    print(f"\n🧪 Calculating descriptors for molecule...")
    
    # Calculate descriptors
    desc_df = calculate_descriptors_from_smiles(smiles)
    
    # Extract features
    X = extract_features(desc_df, model_features)
    
    # Print feature summary
    print_feature_summary(X, desc_df, model_features)

# =====================================================
# PREDICTION PIPELINE
# =====================================================
def predict_smiles(smiles: str):
    """Run the complete prediction pipeline."""
    print(f"\n🧪 Calculating descriptors for molecule...")
    
    # Calculate descriptors
    desc_df = calculate_descriptors_from_smiles(smiles)
    
    # Extract features
    X = extract_features(desc_df, model_features)
    
    # Print feature summary
    print_feature_summary(X, desc_df, model_features)
    
# =====================================================
# PREDICTION PIPELINE
# =====================================================
def predict_smiles(smiles: str):
    """Run the complete prediction pipeline."""
    print(f"\n🧪 Calculating descriptors for molecule...")
    print(f"SMILES: {smiles}")
    
    # Calculate descriptors
    desc_df = calculate_descriptors_from_smiles(smiles)
    
    # Extract features with improved matching (returns X AND mapping)
    X, feature_mapping = extract_features(desc_df, model_features)
    
    # Print feature summary
    print_feature_summary(X, desc_df, model_features, feature_mapping)
    
    # Make prediction
    pred_class = model.predict(X)[0]
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X)[0]
    else:
        pred_proba = None
    
    return X, pred_class, pred_proba, feature_mapping

# =====================================================
# SHAP EXPLANATION
# =====================================================
def explain_prediction(X, model, feature_mapping, pred_class):
    """Generate SHAP explanation for the prediction."""
    try:
        print("\n🔍 Generating SHAP explanation...")
        
        # Determine explainer type based on model
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Handle binary classification list output
            if isinstance(shap_values, list):
                shap_array = shap_values[1] if pred_class == 1 else shap_values[0]
                # For waterfall, we usually want the Explainer object, not just values
                # Re-instantiate generic explainer for better plot support
                explainer_gen = shap.Explainer(model, X)
                shap_obj = explainer_gen(X)
            else:
                shap_obj = shap.Explanation(shap_values, base_values=explainer.expected_value, data=X)
        else:
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(X)
        
        # --- Visualization ---
        plt.figure(figsize=(16, 6))
        
        # 1. Waterfall Plot (Local explanation)
        try:
            plt.subplot(1, 2, 1)
            # If shap_obj is list (older shap versions), use the specific class index
            if isinstance(shap_obj, list): 
                shap_to_plot = shap_obj[0] 
            elif len(shap_obj.shape) == 3: # (samples, features, classes)
                shap_to_plot = shap_obj[0, :, pred_class]
            else:
                shap_to_plot = shap_obj[0]
                
            shap.plots.waterfall(shap_to_plot, max_display=15, show=False)
            plt.title("SHAP Waterfall (Local Explanation)", fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"⚠️ Waterfall plot skipped: {e}")

        # 2. Bar Plot (Global/Local importance)
        try:
            plt.subplot(1, 2, 2)
            shap.plots.bar(shap_to_plot, max_display=15, show=False)
            plt.title("Feature Importance", fontsize=12, fontweight='bold')
        except Exception as e:
             print(f"⚠️ Bar plot skipped: {e}")

        plt.tight_layout()
        plt.show()
        return True
        
    except Exception as e:
        print(f"⚠️ Could not generate complete SHAP explanation: {e}")
        # Fallback: Simple Feature Importance from inputs
        print("Showing raw feature values instead.")
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Value': X.iloc[0].values
        })
        # Sort by absolute value to show most impactful features (proxy)
        feature_importance['Absolute'] = np.abs(feature_importance['Value'])
        feature_importance = feature_importance.sort_values('Absolute', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Value'])
        plt.xlabel('Feature Value')
        plt.title('Top 15 Feature Values (Fallback)')
        plt.tight_layout()
        plt.show()
        return True

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 MOLECULE PREDICTION SYSTEM")
    print("="*60)
    
    # Example molecules
    example_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",      # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
    ]
    
    print("\n💊 Example molecules:")
    for i, sm in enumerate(example_smiles, 1):
        print(f"  {i}. {sm}")
    
    # Get Input
    smiles_input = input("\n🔬 Enter SMILES string (or press Enter for Aspirin): ").strip()
    
    if not smiles_input:
        smiles_input = example_smiles[0]
        print(f"Using default: {smiles_input}")
    
    try:
        # Run prediction
        X_input, prediction, probability, mapping = predict_smiles(smiles_input)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULTS")
        print("="*60)
        
        class_names = {0: "Inactive/Negative", 1: "Active/Positive"}
        pred_label = class_names.get(prediction, f"Class {prediction}")
        
        print(f"\n📋 Prediction: {prediction} ({pred_label})")
        
        if probability is not None:
            print(f"\n📊 Confidence Scores:")
            for class_idx, prob in enumerate(probability):
                class_name = class_names.get(class_idx, f"Class {class_idx}")
                print(f"   • {class_name}: {prob:.1%}")
            
            # Interpret confidence
            confidence = max(probability)
            if confidence > 0.8:
                conf_level = "High"
            elif confidence > 0.6:
                conf_level = "Moderate"
            else:
                conf_level = "Low"
            print(f"\n   💪 Confidence Level: {conf_level} ({confidence:.1%})")
        
        # Generate explanation
        print("\n" + "="*60)
        print("🔍 MODEL EXPLANATION")
        print("="*60)
        
        explain_prediction(X_input, model, mapping, prediction)
        
        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except ValueError as e:
        print(f"\n❌ Input Error: {e}")
        print("Please check your SMILES string syntax.")
    except Exception as e:
        print(f"\n❌ System Error: {e}")
        traceback.print_exc()
