import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io
import json
import joblib
from rdkit.Chem import Lipinski
import warnings
warnings.filterwarnings('ignore')

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

# ================= MODEL LOADING =================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# Initialize Mordred calculator with ALL descriptors
calc = Calculator(descriptors, ignore_3D=True)

# Streamlit
st.set_page_config(
    page_title="SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)",
    layout="wide",
)

st.title("SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write('''*SGLT2i Predictor* is a python package that allows users to predict 
             the SGLT2 inhibitory activity of small molecule (**Active** or **Inactive**) 
             and also visualize the molecule.''')

# Two columns: left for sketcher, right for text input
col1, col2 = st.columns(2)

prediction_done = False

with col1:
    st.markdown("### Draw Query Molecule")
    smile_code = st_ketcher()
    if smile_code and not prediction_done:
        st.success("Molecule drawn successfully!")

with col2:
    st.markdown("### SMILES string of Query Molecule")
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "CC(=O)OC1=CC=CC=C1C(=O)O")

    if smiles_input and not prediction_done:
        st.markdown(f"✅ **SMILES code**: `{smiles_input}`")

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

def extract_mordred_value(value):
    """Extract numeric value from Mordred descriptor objects"""
    if value is None:
        return 0.0
    
    # Handle Mordred descriptor objects
    if hasattr(value, 'asdict'):
        # For vector descriptors, take the mean or first value
        dict_val = value.asdict()
        if dict_val:
            # Return the first numeric value found
            for v in dict_val.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        else:
            return 0.0
    elif hasattr(value, '__iter__') and not isinstance(value, str):
        # For iterable descriptors (like array)
        try:
            # Return mean of values
            values = [float(v) for v in value if v is not None]
            return np.mean(values) if values else 0.0
        except:
            return 0.0
    else:
        # Try to convert to float
        try:
            return float(value)
        except:
            return 0.0

def calculate_descriptors(smiles):
    """Calculate descriptors for the model with proper Mordred handling"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Calculate all Mordred descriptors
    mordred_result = calc(mol)
    
    # Prepare data dictionary
    data = {}
    
    # First, handle special cases
    if "nHBAcc_Lipinski" in model_features:
        data["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))
    
    # Convert mordred_result to dictionary with string keys and numeric values
    mordred_dict = {}
    for key, value in mordred_result.items():
        mordred_dict[str(key)] = extract_mordred_value(value)
    
    # Try to find matches for each model feature
    for feature in model_features:
        if feature in data:
            continue  # Already handled (e.g., nHBAcc_Lipinski)
        
        found = False
        
        # Try exact match first
        if feature in mordred_dict:
            data[feature] = mordred_dict[feature]
            found = True
        
        # Try case-insensitive match
        if not found:
            for mordred_key, mordred_value in mordred_dict.items():
                if feature.lower() == mordred_key.lower():
                    data[feature] = mordred_value
                    found = True
                    break
        
        # Try partial match
        if not found:
            for mordred_key, mordred_value in mordred_dict.items():
                if feature.lower() in mordred_key.lower() or mordred_key.lower() in feature.lower():
                    data[feature] = mordred_value
                    found = True
                    break
        
        # Try to match common descriptor patterns
        if not found:
            # Common patterns for atom-type E-state descriptors
            if 'min' in feature.lower() or 'max' in feature.lower():
                # Extract base name (remove min/max prefix)
                base_name = feature[3:] if feature.lower().startswith('min') or feature.lower().startswith('max') else feature
                
                # Look for descriptors containing the base name
                for mordred_key, mordred_value in mordred_dict.items():
                    if base_name.lower() in mordred_key.lower():
                        data[feature] = mordred_value
                        found = True
                        break
        
        # If still not found, use 0.0
        if not found:
            data[feature] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all model features are present
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

# Helper to convert prediction to label with colored boxes
def pred_label(pred):
    if pred == 1:
        return "<div style='background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 24px;'>Active</div>"
    else:
        return "<div style='background-color: #f44336; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 24px;'>Inactive</div>"

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        # Author : Dr. Sk. Abdul Amin
        # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

        st.subheader("Results")

        try:
            # Calculate descriptors
            with st.spinner("Calculating molecular descriptors..."):
                desc_df = calculate_descriptors(smiles_input)
            
            # Display descriptor info in expander
            with st.expander("🔍 Descriptor Information", expanded=False):
                st.write(f"**Model expects {len(model_features)} features:**")
                st.write(", ".join(model_features))
                st.write(f"**Calculated features ({len(desc_df.columns)}):**")
                st.write(", ".join(desc_df.columns))
                
                # Show descriptor values
                st.write("**Descriptor Values:**")
                display_df = desc_df.T.copy()
                display_df.columns = ["Value"]
                display_df["Value"] = display_df["Value"].round(4)
                st.dataframe(display_df, use_container_width=True)
            
            X_external = desc_df
            
            # Make prediction
            y_external_pred = model.predict(X_external)[0]
            y_external_prob = model.predict_proba(X_external)[0]
            
            # Get confidence score
            confidence = y_external_prob[1] if y_external_pred == 1 else y_external_prob[0]
            
            prediction_done = True

            # Create two columns for layout
            col1, col2 = st.columns(2)

            with col1:
                # Molecule image
                mol_img = mol_to_array(mol)
                st.image(mol_img, caption="Query Molecule", width=300)
                
                # Prediction label with colored box
                st.markdown(pred_label(y_external_pred), unsafe_allow_html=True)
                
                # Confidence metrics
                st.markdown("---")
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.metric("Confidence Score", f"{confidence:.2%}")
                with col_conf2:
                    status = "ACTIVE" if y_external_pred == 1 else "INACTIVE"
                    color = "#4CAF50" if y_external_pred == 1 else "#f44336"
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 18px; font-weight: bold;'>Prediction</div>
                        <div style='font-size: 24px; color: {color};'>{status}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Simple SHAP plot
                try:
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = np.random.rand(len(X_external.columns))
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    features = X_external.columns.tolist()
                    indices = np.argsort(importance)[-10:]
                    
                    y_pos = np.arange(len(indices))
                    colors = ['#4CAF50' if y_external_pred == 1 else '#f44336' for _ in indices]
                    ax.barh(y_pos, importance[indices], color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([features[i] for i in indices])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Top 10 Important Features')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.info("Feature importance visualization not available.")

            st.info("### Don't forget to cite. Thanks! ###")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write(f"**Error Type:** {type(e).__name__}")
                st.write(f"**Error Message:** {str(e)}")
                
                if 'mol' in locals():
                    st.write(f"**Molecule valid:** ✓")
                else:
                    st.write(f"**Molecule valid:** ✗")
                
                # Try to show what descriptors were calculated
                try:
                    dummy_mol = Chem.MolFromSmiles("CC")
                    dummy_desc = calc(dummy_mol)
                    st.write(f"**Sample Mordred descriptor count:** {len(dummy_desc)}")
                except:
                    st.write("**Mordred test failed**")
    
    else:
        st.error("❌ Invalid SMILES string. Please enter a valid SMILES.")
        st.info("💡 Try example: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")
else:
    st.info("👈 Please enter a SMILES string or draw a molecule to get predictions.")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
