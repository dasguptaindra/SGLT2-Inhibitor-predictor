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

calc = Calculator(descriptors, ignore_3D=True)

# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(300, 300)) if mol else None

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mordred_vals = calc(mol)
    data = {}
    for f in model_features:
        try:
            data[f] = float(mordred_vals[f])
        except:
            data[f] = 0.0
    data["nHBAcc_Lipinski"] = Lipinski.NumHAcceptors(mol)
    return pd.DataFrame([data]).fillna(0)

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
    smiles_drawn = st_ketcher() if KETCHER_AVAILABLE else ""

with col2:
    st.subheader("🧬 SMILES Input")
    smiles = st.text_input(
        "Enter or edit SMILES",
        value=smiles_drawn if smiles_drawn else ""
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

desc_df = calculate_descriptors(smiles)
desc_df = desc_df.replace([np.inf, -np.inf], 0)

pred = model.predict(desc_df)[0]
prob = model.predict_proba(desc_df)[0][1]

col1, col2 = st.columns(2)

with col1:
    st.image(draw_molecule(smiles), caption="Query Molecule", width=250)
    if pred == 1:
        st.success("🟢 **ACTIVE – SGLT2 Inhibitor**")
    else:
        st.error("🔴 **INACTIVE – Non-Inhibitor**")

    st.metric("Confidence Score", f"{prob:.2%}")

with col2:
    st.subheader("📈 SHAP Interpretation")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(desc_df)

    shap_val = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

    fig, ax = plt.subplots(figsize=(5, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_val[:10],
            base_values=base_val,
            data=desc_df.iloc[0].values[:10],
            feature_names=desc_df.columns[:10]
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

# ================= DESCRIPTORS =================
with st.expander("🔬 Calculated Descriptors"):
    st.dataframe(desc_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center>🧪 <b>SGLT2i Prediction Tool</b> | Built with Streamlit</center>",
    unsafe_allow_html=True
)
