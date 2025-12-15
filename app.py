# =========================
# SGLT2i Predictor
# =========================

import os, json, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------- RDKit & Mordred --------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski
from mordred import Calculator, descriptors
from mordred.AtomTypeEState import AtomTypeEState
import shap

# -------- Optional Ketcher --------
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except:
    KETCHER_AVAILABLE = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SGLT2i Predictor",
    page_icon="💊",
    layout="wide"
)

# ================= COMPACT CSS =================
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1, h2, h3 { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ================= MODEL LOADING =================
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# Initialize Mordred calculator with ALL descriptors
calc = Calculator(descriptors, ignore_3D=True)

# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(300, 300)) if mol else None

def get_mordred_descriptor_names():
    """Get all available Mordred descriptor names"""
    # Create a temporary molecule to calculate descriptors
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

# ================= HEADER =================
st.title("SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write(
        "**SGLT2i Predictor** allows users to predict the SGLT2 inhibitory activity of small molecules/drug molecules "
        "using a machine learning model and provides SHAP-based interpretability."
    )

# ================= INPUT SECTION =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Draw Molecule")
    if KETCHER_AVAILABLE:
        smiles_drawn = st_ketcher()
    else:
        smiles_drawn = ""
        st.info("Ketcher not available. Please enter SMILES manually.")

with col2:
    st.subheader("🧬 SMILES Input")
    smiles = st.text_input(
        "Enter or edit SMILES",
        value=smiles_drawn if smiles_drawn else "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin as example
    )

# ================= VALIDATION =================
if not smiles:
    st.info("Please draw a molecule or enter a SMILES string.")
    st.stop()

if not validate_smiles(smiles):
    st.error("Invalid SMILES string.")
    st.stop()

# ================= RESULTS =================
st.markdown("---")
st.subheader("📊 Results")

try:
    desc_df = calculate_descriptors(smiles)
    
    # Display descriptor debugging info
    with st.expander("🔍 Descriptor Debug Info"):
        st.write(f"Model expects {len(model_features)} features:")
        st.write(model_features)
        st.write(f"Calculated {len(desc_df.columns)} features:")
        st.write(list(desc_df.columns))
        
        # Check for missing features
        missing_features = set(model_features) - set(desc_df.columns)
        if missing_features:
            st.warning(f"Missing features: {missing_features}")
    
    # Make prediction
    pred = model.predict(desc_df)[0]
    prob = model.predict_proba(desc_df)[0][1]
    
    col1, col2 = st.columns(2)

    with col1:
        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Query Molecule", width=250)
        
        if pred == 1:
            st.success("🟢 **ACTIVE – SGLT2 Inhibitor**")
        else:
            st.error("🔴 **INACTIVE – Non-Inhibitor**")

        st.metric("Confidence Score", f"{prob:.2%}")

    with col2:
        st.subheader("📈 SHAP Interpretation")
        
        # Create SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(desc_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]  # For binary classification
            base_val = explainer.expected_value[1]
        else:
            shap_val = shap_values[0]
            base_val = explainer.expected_value
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=desc_df.iloc[0].values,
            feature_names=desc_df.columns
        )
        
        # Plot top 10 features
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Additional SHAP summary
        with st.expander("📋 SHAP Summary"):
            st.write("Feature contributions to prediction:")
            feature_contributions = pd.DataFrame({
                'Feature': desc_df.columns,
                'SHAP Value': shap_val,
                'Feature Value': desc_df.iloc[0].values
            })
            feature_contributions['Absolute Impact'] = np.abs(feature_contributions['SHAP Value'])
            feature_contributions = feature_contributions.sort_values('Absolute Impact', ascending=False)
            st.dataframe(feature_contributions.head(15), use_container_width=True)

except Exception as e:
    st.error(f"Error calculating descriptors or making prediction: {str(e)}")
    st.info("Please try a different molecule or check the SMILES string.")
    st.stop()

# ================= DESCRIPTORS =================
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

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    """
    <center>
    🧪 <b>SGLT2i Prediction Tool</b> | Built with Streamlit<br>
    <small>For research use only. Not for clinical decision making.</small>
    </center>
    """,
    unsafe_allow_html=True
)

# ================= SIDEBAR INFO =================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool predicts SGLT2 inhibitory activity using:
    - **Machine Learning**: Gradient Boosting model
    - **Descriptors**: 13 key molecular descriptors
    - **Interpretability**: SHAP-based feature importance
    
    **Model Features:**
    1. MAXaaN - Maximum aaN atom-type E-state
    2. MINaaN - Minimum aaN atom-type E-state
    3. nN - Number of Nitrogen atoms
    4. nFARing - Number of fused aromatic rings
    5. naHRing - Number of aliphatic heterocycles
    6. MAXsCl - Maximum sCl atom-type E-state
    7. NaaN - Number of aaN atoms
    8. nHBAcc_Lipinski - Hydrogen bond acceptors (Lipinski)
    9. BCUTs-1h - BCUT descriptor
    10. nFAHRing - Number of fused aromatic heterocycles
    11. ATSC2c - Autocorrelation descriptor
    12. MDEC-33 - Molecular distance edge descriptor
    13. MATS2c - Moran autocorrelation descriptor
    """)
    
    st.info("💡 **Tip**: Try common SGLT2 inhibitors like Dapagliflozin (CCCCCC1=CC(=C(C(=C1)C2C(C(C(O2)CO)O)O)OC3C(C(C(C(O3)CO)O)O)O)O)OC4C(C(C(C(O4)CO)O)O)O)")
