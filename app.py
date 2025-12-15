# =====================================================
# External Molecule Predictor
# =====================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    model_features = json.load(f)

print(f"\n✅ Loaded model with {len(model_features)} features")

# =====================================================
# DESCRIPTOR NAME MAPPING & PATTERN MATCHING
# =====================================================
def find_matching_descriptor(mordred_keys, target_name):
    """
    Find the best matching descriptor name from Mordred keys.
    Handles case variations and naming differences.
    """
    target_lower = target_name.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    # First, try exact match (case-insensitive)
    for mordred_key in mordred_keys:
        if str(mordred_key).lower() == target_name.lower():
            return str(mordred_key)
    
    # Try matching without special characters
    for mordred_key in mordred_keys:
        mordred_clean = str(mordred_key).lower().replace('-', '').replace('_', '').replace(' ', '')
        if mordred_clean == target_lower:
            return str(mordred_key)
    
    # Try substring match
    for mordred_key in mordred_keys:
        if target_name.lower() in str(mordred_key).lower():
            return str(mordred_key)
    
    # Try partial match for patterns like MINaaN -> MinAaN
    for mordred_key in mordred_keys:
        mordred_lower = str(mordred_key).lower()
        # Check if target appears in descriptor name (allowing for variations)
        if any(part in mordred_lower for part in target_lower.split('_')):
            return str(mordred_key)
    
    return None

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
# FEATURE EXTRACTION WITH IMPROVED MATCHING
# =====================================================
def extract_features(desc_df, required_features):
    """Extract required features from Mordred descriptors using pattern matching."""
    X = pd.DataFrame(0.0, index=[0], columns=required_features)
    available_columns = [str(c) for c in desc_df.columns]
    
    print(f"\n📊 Available Mordred descriptors: {len(available_columns)}")
    print(f"🔍 Looking for {len(required_features)} required features...")
    
    # Create a mapping of feature names to Mordred descriptor names
    feature_mapping = {}
    
    for feat in required_features:
        # Try to find the matching descriptor in Mordred output
        matched_name = find_matching_descriptor(available_columns, feat)
        
        if matched_name:
            feature_mapping[feat] = matched_name
            val = desc_df.at[0, matched_name]
            
            if pd.notna(val) and val not in [np.inf, -np.inf]:
                X.at[0, feat] = float(val)
                # print(f"✅ Found: '{feat}' -> '{matched_name}' = {val}")
            else:
                X.at[0, feat] = 0.0  # Default for NaN/inf values
                # print(f"⚠️  Found but invalid: '{feat}' -> '{matched_name}' (NaN/Inf)")
        else:
            # Check common variations
            common_variations = {
                'MINaaN': ['MinAaN', 'minaann', 'MinAaN'],
                'MAXaaN': ['MaxAaN', 'maxaann', 'MaxAaN'],
                'nHBAcc_Lipinski': ['nHBAcc', 'nhbacc', 'nHBAcc2'],
                'BCUTs-1h': ['BCUTw-1h', 'BCUTc-1h', 'BCUTp-1h', 'bcut'],
            }
            
            found_variant = None
            if feat in common_variations:
                for variant in common_variations[feat]:
                    if variant in available_columns:
                        found_variant = variant
                        break
            
            if found_variant:
                feature_mapping[feat] = found_variant
                val = desc_df.at[0, found_variant]
                if pd.notna(val) and val not in [np.inf, -np.inf]:
                    X.at[0, feat] = float(val)
                    print(f"🔄 Using variant: '{feat}' -> '{found_variant}'")
                else:
                    X.at[0, feat] = 0.0
            else:
                # Try to find any descriptor containing the feature name
                for col in available_columns:
                    if feat.lower().replace('-', '').replace('_', '') in col.lower().replace('-', '').replace('_', ''):
                        feature_mapping[feat] = col
                        val = desc_df.at[0, col]
                        if pd.notna(val) and val not in [np.inf, -np.inf]:
                            X.at[0, feat] = float(val)
                            print(f"🔍 Partial match: '{feat}' -> '{col}'")
                        break
                else:
                    # Feature not found at all
                    X.at[0, feat] = 0.0
                    print(f"❌ Not found: '{feat}'")
    
    # Print summary of found features
    found_count = (X != 0).sum().sum()
    print(f"\n📈 Successfully mapped {found_count}/{len(required_features)} features")
    
    return X, feature_mapping

# =====================================================
# FEATURE SUMMARY
# =====================================================
def print_feature_summary(X, desc_df, model_features, feature_mapping):
    """Print a summary of feature values with mapping information."""
    available_columns = [str(c) for c in desc_df.columns]
    
    print("\n" + "="*60)
    print("FEATURE VALUE SUMMARY")
    print("="*60)
    
    print(f"{'Feature Name':<25} {'Mordred Name':<25} {'Value':<15} Status")
    print("-" * 85)
    
    for feat in model_features:
        mordred_name = feature_mapping.get(feat, "Not found")
        value = X.at[0, feat]
        
        if mordred_name in available_columns:
            original_val = desc_df.at[0, mordred_name]
            if pd.notna(original_val) and original_val not in [np.inf, -np.inf]:
                status = "✅ Valid"
                color_code = ""
            else:
                status = "⚠️  NaN/Inf"
                color_code = ""
        else:
            status = "❌ Not in Mordred"
            color_code = ""
        
        print(f"{feat:<25} {mordred_name:<25} {value:<15.6f} {status}")
    
    non_zero_count = (X != 0).sum(axis=1).iloc[0]
    zero_count = len(model_features) - non_zero_count
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(f"📊 Total features required: {len(model_features)}")
    print(f"✅ Features with non-zero values: {non_zero_count}")
    print(f"⚠️  Features with zero/default values: {zero_count}")
    
    if zero_count > 0:
        print("\nFeatures with zero values:")
        for feat in model_features:
            if X.at[0, feat] == 0:
                print(f"  • {feat}")

# =====================================================
# PREDICTION PIPELINE
# =====================================================
def predict_smiles(smiles: str):
    """Run the complete prediction pipeline."""
    print(f"\n🧪 Calculating descriptors for molecule...")
    print(f"SMILES: {smiles}")
    
    # Calculate descriptors
    desc_df = calculate_descriptors_from_smiles(smiles)
    
    # Extract features with improved matching
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
def explain_prediction(X, model, feature_mapping):
    """Generate SHAP explanation for the prediction."""
    try:
        print("\n🔍 Generating SHAP explanation...")
        
        # Create a simplified explainer
        if hasattr(model, 'predict_proba'):
            # For classifiers
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Get the appropriate SHAP values for the predicted class
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Binary classification
                shap_array = shap_values[1] if pred_class == 1 else shap_values[0]
            else:
                shap_array = shap_values
        else:
            # For regressors
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            shap_array = shap_values.values
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Waterfall plot
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        ax1 = plt.gca()
        ax1.set_title("SHAP Waterfall Plot", fontsize=14, fontweight='bold')
        
        # Bar plot for feature importance
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=15, show=False)
        ax2 = plt.gca()
        ax2.set_title("Feature Importance", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return True
        
    except Exception as e:
        print(f"⚠️  Could not generate SHAP explanation: {e}")
        print("Trying alternative explanation method...")
        
        # Try simpler explanation
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.show()
            return True
        except:
            print("SHAP explanation failed. Showing feature importance instead.")
            
            # Show feature values as fallback
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Value': X.iloc[0].values
            })
            feature_importance['Absolute'] = np.abs(feature_importance['Value'])
            feature_importance = feature_importance.sort_values('Absolute', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(feature_importance['Feature'], feature_importance['Value'])
            plt.xlabel('Feature Value')
            plt.title('Top 10 Feature Values (Fallback)')
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
    
    # Example SMILES or get from user
    example_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    print("\n💊 Example molecules:")
    for i, sm in enumerate(example_smiles, 1):
        print(f"  {i}. {sm}")
    
    # Get SMILES input
    smiles_input = input("\n🔬 Enter SMILES string (or press Enter for aspirin): ").strip()
    
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
        
        explain_prediction(X_input, model, mapping)
        
        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please check your SMILES string and try again.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
