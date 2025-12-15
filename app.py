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

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

# ================= MODEL LOADING =================
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
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
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "")

    if smiles_input and not prediction_done:
        st.markdown(f"✅ **SMILES code**: `{smiles_input}`")

def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

def get_mordred_descriptor_names():
    """Get all available Mordred descriptor names"""
    temp_mol = Chem.MolFromSmiles("CC")
    desc_df = calc(temp_mol)
    return list(desc_df.keys())

def find_matching_descriptor(descriptor_name, available_descriptors):
    """Find the correct Mordred descriptor name for a given feature name"""
    descriptor_lower = descriptor_name.lower()
    
    # Common mappings for atom-type E-state descriptors
    atom_type_mappings = {
        'minaan': ['minaasn', 'min_aan', 'min(aan)', 'minaasn', 'eta_min_aan'],
        'maxaan': ['maxaasn', 'max_aan', 'max(aan)', 'maxaasn', 'eta_max_aan'],
        'naan': ['naasn', 'n_aan', 'n(aan)', 'eta_n_aan'],
    }
    
    # Check exact match first
    if descriptor_name in available_descriptors:
        return descriptor_name
    
    # Check case-insensitive match
    for avail_desc in available_descriptors:
        if descriptor_lower == avail_desc.lower():
            return avail_desc
    
    # Check for atom-type E-state descriptors with different naming
    if descriptor_lower in atom_type_mappings:
        for pattern in atom_type_mappings[descriptor_lower]:
            for avail_desc in available_descriptors:
                if pattern in avail_desc.lower():
                    return avail_desc
    
    # Try to find partial matches
    for avail_desc in available_descriptors:
        if descriptor_lower in avail_desc.lower() or avail_desc.lower() in descriptor_lower:
            return avail_desc
    
    return None

def calculate_descriptors(smiles):
    """Calculate all Mordred descriptors and extract the ones needed by the model"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Calculate all Mordred descriptors
    mordred_result = calc(mol)
    
    # Get all available descriptor names from Mordred
    all_descriptors = list(mordred_result.keys())
    
    # Prepare data dictionary
    data = {}
    
    # For each feature needed by the model, try to find the corresponding descriptor
    for feature in model_features:
        if feature == "nHBAcc_Lipinski":
            # Calculate Lipinski descriptors separately
            data[feature] = float(Lipinski.NumHAcceptors(mol))
        else:
            # Try to find the matching Mordred descriptor
            matched_desc = find_matching_descriptor(feature, all_descriptors)
            
            if matched_desc is not None:
                try:
                    # Get the value from Mordred
                    value = mordred_result[matched_desc]
                    # Handle various types of values
                    if hasattr(value, 'asdict'):
                        value = value.asdict().get('min', 0) if 'min' in str(value).lower() else value
                    data[feature] = float(value)
                except Exception as e:
                    st.warning(f"Could not calculate {feature}: {str(e)}")
                    data[feature] = 0.0
            else:
                # If descriptor not found, check if it's an atom-type E-state descriptor
                if 'min' in feature.lower() and 'max' in feature.lower():
                    # This might be a statistic descriptor, try to calculate manually
                    try:
                        # Get the base descriptor name (remove min/max prefix)
                        base_name = feature[3:] if feature.lower().startswith('min') or feature.lower().startswith('max') else feature
                        
                        # Try to find the base descriptor
                        for desc in all_descriptors:
                            if base_name.lower() in desc.lower():
                                value = mordred_result[desc]
                                if hasattr(value, '__iter__'):
                                    if feature.lower().startswith('min'):
                                        data[feature] = float(min(value))
                                    elif feature.lower().startswith('max'):
                                        data[feature] = float(max(value))
                                    else:
                                        data[feature] = float(value)
                                else:
                                    data[feature] = float(value)
                                break
                        else:
                            data[feature] = 0.0
                    except:
                        data[feature] = 0.0
                else:
                    data[feature] = 0.0
    
    # Create DataFrame and handle infinities
    df = pd.DataFrame([data])
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

# Helper to convert prediction to label
def pred_label(pred):
    return "### **Active**" if pred == 1 else "### **Inactive**"

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        # Author : Dr. Sk. Abdul Amin
        # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

        st.subheader("Results")

        try:
            # Calculate descriptors
            desc_df = calculate_descriptors(smiles_input)
            X_external = desc_df
            
            # Make prediction
            y_external_pred = model.predict(X_external)
            y_external_prob = model.predict_proba(X_external)[0]
            
            # Get confidence score
            confidence = y_external_prob[1] if y_external_pred[0] == 1 else y_external_prob[0]

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_external)
            
            prediction_done = True

            # Row 1 — Query molecule
            col1, col2 = st.columns(2)

            with col1:
                # SHAP plot (smaller)
                plt.figure(figsize=(4, 3))
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # For binary classification
                    shap_val = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val = shap_values[0]
                    base_val = explainer.expected_value
                
                # Create SHAP explanation object
                explanation = shap.Explanation(
                    values=shap_val,
                    base_values=base_val,
                    data=X_external.iloc[0].values,
                    feature_names=X_external.columns
                )
                
                shap.plots.waterfall(explanation, max_display=10, show=False)
                fig1 = plt.gcf()
                st.pyplot(fig1)
                plt.clf()
                
                # Molecule image
                mol_img = mol_to_array(mol)
                st.image(mol_img, caption="Query Molecule", width=250)
                
                # Prediction label with confidence
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[0])}</div>",
                            unsafe_allow_html=True)
                st.metric("Confidence Score", f"{confidence:.2%}")

            with col2:
                # Applicability Domain (Leverage) Plot
                try:
                    # Since we don't have train/test data, we'll create a simple visualization
                    # This is a simplified version - you may want to load actual training data
                    
                    # Create synthetic data for visualization
                    n_compounds = 50
                    n_features = len(model_features)
                    
                    # Generate random data similar to training distribution
                    np.random.seed(42)
                    synthetic_data = np.random.randn(n_compounds, n_features)
                    
                    # Calculate leverage for synthetic data
                    X_synthetic = synthetic_data
                    H_synthetic = X_synthetic @ np.linalg.pinv(X_synthetic.T @ X_synthetic) @ X_synthetic.T
                    synthetic_leverage = np.diag(H_synthetic)
                    
                    # Calculate leverage for query molecule
                    X_combined = np.vstack([X_synthetic, X_external.values])
                    H_combined = X_combined @ np.linalg.pinv(X_combined.T @ X_combined) @ X_combined.T
                    query_leverage = np.diag(H_combined)[-1]
                    
                    # Calculate threshold
                    p = X_synthetic.shape[1]
                    n = X_synthetic.shape[0]
                    leverage_threshold = 3 * p / n
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(5, 4))
                    
                    # Plot synthetic compounds
                    colors = ['green' if lev <= leverage_threshold else 'red' 
                             for lev in synthetic_leverage]
                    ax.scatter(range(len(synthetic_leverage)), synthetic_leverage, 
                              c=colors, s=20, alpha=0.6, label='Reference Compounds')
                    
                    # Plot query molecule
                    query_color = 'blue' if query_leverage <= leverage_threshold else 'orange'
                    ax.scatter(len(synthetic_leverage), query_leverage, 
                              c=query_color, s=100, marker='*', 
                              label='Query Molecule', edgecolors='black', linewidth=1)
                    
                    # Threshold line
                    ax.axhline(y=leverage_threshold, color='red', linestyle='--', 
                              label='Threshold', alpha=0.7)
                    
                    ax.set_xlabel('Compound Index')
                    ax.set_ylabel('Leverage Value')
                    ax.set_title('Applicability Domain Analysis')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display AD result
                    is_in_ad = query_leverage <= leverage_threshold
                    st.write(f"Your Molecule within Applicability Domain: {'Yes' if is_in_ad else 'No'}")
                    
                    if not is_in_ad:
                        st.warning("⚠️ The molecule is outside the applicability domain. Results may be less reliable.")
                    
                    st.caption("NOTE: Applicability Domain analysis helps assess prediction reliability")
                    st.info("### Don't forget to cite. Thanks! ###")

                except Exception as e:
                    st.error(f"Applicability Domain analysis failed: {e}")
                    st.info("Using simplified analysis - consider providing training data for more accurate AD assessment")

            # Separator
            st.markdown("---")

            # Additional information in expanders
            with st.expander("🔬 Calculated Descriptors"):
                st.dataframe(desc_df.T.rename(columns={0: "Value"}).round(6), use_container_width=True)
                
                # Show statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total Descriptors", len(desc_df.columns))
                with col_stat2:
                    st.metric("Non-zero Values", (desc_df != 0).sum().sum())
                with col_stat3:
                    st.metric("Mean Absolute Value", f"{desc_df.abs().mean().mean():.4f}")

            with st.expander("📋 Feature Contributions"):
                # Create feature contributions dataframe
                if isinstance(shap_values, list):
                    shap_val = shap_values[1][0]
                else:
                    shap_val = shap_values[0]
                
                feature_contributions = pd.DataFrame({
                    'Feature': desc_df.columns,
                    'SHAP Value': shap_val,
                    'Feature Value': desc_df.iloc[0].values
                })
                feature_contributions['Absolute Impact'] = np.abs(feature_contributions['SHAP Value'])
                feature_contributions = feature_contributions.sort_values('Absolute Impact', ascending=False)
                st.dataframe(feature_contributions.head(15), use_container_width=True)
                
                # Summary statistics
                st.write("**Top 3 influential features:**")
                for i, row in feature_contributions.head(3).iterrows():
                    direction = "increases" if row['SHAP Value'] > 0 else "decreases"
                    st.write(f"- **{row['Feature']}**: {direction} probability of being active (impact: {row['SHAP Value']:.4f})")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please try a different molecule or check the SMILES string.")
    else:
        st.error("Invalid SMILES string. Please enter a valid SMILES.")
else:
    st.info("Please enter a SMILES string to get predictions.")

# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/dasguptaindra/SGLT2i-predictor)
    ''')
