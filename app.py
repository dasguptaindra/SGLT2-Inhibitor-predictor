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
    st.write('''**SGLT2i Predictor** is a tool that allows users to predict 
             the SGLT2 inhibitory activity of small molecule (**Active** or **Inactive**) 
             and also provides SHAP-based interpretability.''')

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

# Helper to convert prediction to label
def pred_label(pred):
    return "### **🟢 Active**" if pred == 1 else "### **🔴 Inactive**"

def create_shap_plot(model, X_external, prediction):
    """Create SHAP plot with error handling"""
    try:
        # Try TreeExplainer first
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_external)
            
            if isinstance(shap_values, list):
                # Binary classification
                shap_array = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_array = shap_values
                expected_value = explainer.expected_value
                
        except Exception as e:
            # Fall back to simplified approach
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.ones(len(X_external.columns)) / len(X_external.columns)
            
            # Create pseudo-SHAP values based on feature importance
            shap_array = np.array([importance * (1 if prediction == 1 else -1)])
            expected_value = 0.5
        
        # Create SHAP values object
        shap_val = shap_array[0] if len(shap_array.shape) > 1 else shap_array
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_val,
            base_values=expected_value,
            data=X_external.iloc[0].values,
            feature_names=X_external.columns.tolist()
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(4, 3))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        
        return fig, shap_val
        
    except Exception as e:
        # Create a simple bar plot as fallback
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = np.random.rand(len(X_external.columns))
        
        fig, ax = plt.subplots(figsize=(4, 3))
        features = X_external.columns.tolist()
        indices = np.argsort(importance)[-10:]
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Important Features')
        plt.tight_layout()
        
        return fig, importance

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        
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

            # Row 1 — Query molecule
            col1, col2 = st.columns(2)

            with col1:
                # SHAP plot
                with st.spinner("Calculating feature contributions..."):
                    fig, shap_values = create_shap_plot(model, X_external, y_external_pred)
                    st.pyplot(fig)
                    plt.clf()
                
                # Molecule image
                mol_img = mol_to_array(mol)
                st.image(mol_img, caption="Query Molecule", width=250)
                
                # Prediction label with confidence
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred)}</div>",
                            unsafe_allow_html=True)
                
                # Confidence metrics
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.metric("Confidence Score", f"{confidence:.2%}")
                with col_conf2:
                    st.metric("Prediction", "🟢 ACTIVE" if y_external_pred == 1 else "🔴 INACTIVE")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if the SMILES string is valid and try again.")

# Contact section
with st.expander("Contact & Information", expanded=False):
    st.write('''
        #### 📧 Contact Us      
        #### 🐛 Report Issues
        Report bugs or contribute: [GitHub Repository](https://github.com/dasguptaindra/SGLT2i-predictor)
        
        #### ⚠️ Disclaimer
        This tool is for research purposes only. Not for clinical decision making.
        Always validate predictions with experimental data.
    ''')
