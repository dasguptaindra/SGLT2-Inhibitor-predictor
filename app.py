# =========================
# SGLT2i Predictor
# =========================

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------- RDKit & Mordred --------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski
from mordred import Calculator, descriptors
import shap

# -------- Optional Ketcher --------
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
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

# Mordred calculator (2D only)
calc = Calculator(descriptors, ignore_3D=True)

# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None


def calculate_descriptors(smiles: str):
    """
    Calculate descriptors using Mordred calculator (simplified approach)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    # Create a list with the single molecule
    molecules = [mol]
    
    # Calculate all descriptors using Mordred
    descriptor_df = calc.pandas(molecules)
    
    failed_descriptors = []
    
    # Check which required descriptors are available
    available_features = []
    for feat in model_features:
        if feat == "nHBAcc_Lipinski":
            # Special case: use RDKit for Lipinski descriptor
            available_features.append(feat)
        elif feat in descriptor_df.columns:
            available_features.append(feat)
        else:
            failed_descriptors.append(feat)
    
    # Create the descriptor dataframe
    data = {}
    
    for feat in model_features:
        if feat == "nHBAcc_Lipinski":
            # Calculate Lipinski descriptor separately
            data[feat] = float(Lipinski.NumHAcceptors(mol))
        elif feat in descriptor_df.columns:
            # Get value from Mordred
            val = descriptor_df[feat].iloc[0]
            # Handle NaN/Inf values
            if pd.isna(val) or np.isinf(val):
                data[feat] = 0.0
                if feat not in failed_descriptors:
                    failed_descriptors.append(feat)
            else:
                data[feat] = float(val)
        else:
            # Descriptor not available
            data[feat] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    # Report status
    success_count = len(model_features) - len(failed_descriptors)
    if success_count == len(model_features):
        st.success(f"✅ Successfully calculated all {len(model_features)} descriptors")
    else:
        st.warning(f"⚠️ Calculated {success_count}/{len(model_features)} descriptors")
        if failed_descriptors:
            with st.expander("View failed descriptors"):
                for failed in failed_descriptors:
                    st.write(f"  • {failed}")
    
    return df, failed_descriptors


# ================= HEADER =================
st.title("SGLT2i Predictor v1.0")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write(
        "**SGLT2i Predictor** allows users to predict the SGLT2 inhibitory activity "
        "of small molecules/drug molecules using a Gradient Boosting classifier "
        "and provides SHAP-based interpretability."
    )

# ================= INPUT SECTION =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Draw Molecule")
    smiles_drawn = st_ketcher() if KETCHER_AVAILABLE else ""

with col2:
    st.subheader("🧬 SMILES Input")
    smiles = st.text_input(
        "Enter or edit SMILES",
        value=smiles_drawn if smiles_drawn else "",
        placeholder="e.g., CCO for ethanol"
    )

# ================= VALIDATION =================
if not smiles:
    st.info("Please draw a molecule or enter a SMILES string.")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string.")
    st.stop()

# ================= DESCRIPTOR CALCULATION =================
with st.spinner("Calculating molecular descriptors..."):
    try:
        desc_df, failed_desc = calculate_descriptors(smiles)
    except Exception as e:
        st.error(f"❌ Error calculating descriptors: {str(e)}")
        st.stop()

# ================= PREDICTION =================
with st.spinner("Making prediction..."):
    try:
        pred = model.predict(desc_df)[0]
        prob = model.predict_proba(desc_df)[0][1]
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.stop()

# ================= RESULTS =================
st.markdown("---")
st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    # Display molecule
    img = draw_molecule(smiles)
    if img:
        st.image(img, caption="Query Molecule", width=250)
    
    # Display prediction
    if pred == 1:
        st.success("🟢 **ACTIVE – SGLT2 Inhibitor**")
    else:
        st.error("🔴 **INACTIVE – Non-Inhibitor**")
    
    st.metric("Prediction Confidence", f"{prob:.2%}")
    
    # Display SMILES
    st.code(smiles, language="text")

with col2:
    st.subheader("📈 SHAP Interpretation")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(desc_df)
        
        # Handle binary classifier SHAP values
        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]
            base_val = explainer.expected_value[1]
        else:
            shap_val = shap_values[0]
            base_val = explainer.expected_value
        
        # Select top |SHAP| features
        top_idx = np.argsort(np.abs(shap_val))[-10:]
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(5, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_val[top_idx],
                base_values=base_val,
                data=desc_df.iloc[0, top_idx].values,
                feature_names=desc_df.columns[top_idx].tolist()
            ),
            show=False,
            max_display=10
        )
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {str(e)}")

# ================= DESCRIPTORS =================
with st.expander("🔬 Calculated Descriptors"):
    st.write(f"**Total descriptors calculated:** {len(desc_df.columns)}")
    
    # Format the dataframe nicely
    display_df = desc_df.T.copy()
    display_df.columns = ["Value"]
    display_df["Value"] = display_df["Value"].apply(lambda x: f"{x:.6f}")
    
    # Color code based on calculation success
    def highlight_row(row):
        if row.name in failed_desc:
            return ['background-color: #ffcccc']  # Light red for failed
        else:
            return ['background-color: #ccffcc']  # Light green for successful
    
    # Apply styling
    styled_df = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center>🧪 <b>SGLT2i Prediction Tool</b> | Built with Streamlit</center>",
    unsafe_allow_html=True
)

# ================= SIDEBAR (Optional) =================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool predicts SGLT2 inhibitory activity using:
    - **13 molecular descriptors**
    - **Gradient Boosting classifier**
    - **SHAP for interpretability**
    
    **Descriptors used:**
    1. MAXaaN
    2. MINaaN  
    3. nN
    4. nFARing
    5. naHRing
    6. MAXsCl
    7. NaaN
    8. nHBAcc_Lipinski
    9. BCUTs-1h
    10. nFAHRing
    11. ATSC2c
    12. MDEC-33
    13. MATS2c
    """)
