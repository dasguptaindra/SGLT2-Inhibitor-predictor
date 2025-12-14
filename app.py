# app.py

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------- RDKit & Mordred Imports --------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Lipinski
except Exception as e:
    st.error(f"RDKit import failed: {e}")
    raise

try:
    from mordred import Calculator, descriptors
except Exception as e:
    st.warning(f"Mordred import warning: {e}")
    Calculator = None
    descriptors = None

try:
    import shap
except Exception as e:
    st.warning(f"SHAP import warning: {e}")
    shap = None

# Optional: Streamlit Ketcher
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor (GBM)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- BEAUTIFUL CUSTOM CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1f77b4, #6f42c1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}

.sub-header {
    font-size: 1.4rem;
    color: #555;
    text-align: center;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.input-area {
    background: linear-gradient(135deg, #f8f9ff, #eef2ff);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e0e7ff;
}

.prediction-box {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 600;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.active {
    background: linear-gradient(135deg, #d4edda, #a8e6cf);
    border: 2px solid #28a745;
    color: #155724;
}

.inactive {
    background: linear-gradient(135deg, #f8d7da, #f5b7b1);
    border: 2px solid #dc3545;
    color: #721c24;
}

.stButton > button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 14px 28px;
    border-radius: 14px;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 30px rgba(255, 75, 43, 0.4);
}

.section-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 0.95rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING --------------------
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"Features file not found: {FEATURES_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)

# -------------------- HELPER FUNCTIONS (UNCHANGED) --------------------
def validate_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mordred_dict = {}
    if Calculator and descriptors:
        calc = Calculator(descriptors, ignore_3D=True)
        results = calc(mol)
        for k, v in results.items():
            try:
                mordred_dict[str(k)] = float(v) if v not in [None, np.nan] else 0.0
            except Exception:
                mordred_dict[str(k)] = 0.0

    rdkit_dict = {"nHBAcc_Lipinski": float(Lipinski.NumHAcceptors(mol))}

    feature_values = {}
    for feat in features:
        if feat in mordred_dict:
            feature_values[feat] = mordred_dict[feat]
        elif feat in rdkit_dict:
            feature_values[feat] = rdkit_dict[feat]
        else:
            feature_values[feat] = 0.0

    df = pd.DataFrame([feature_values], columns=features)
    return df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(420, 420)) if mol else None

# -------------------- HEADER --------------------
st.markdown("<div class='main-header'>🧪 SGLT2 Inhibitor Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Machine Learning–Driven Prediction with Explainable AI</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔬 Molecule Input")
input_mode = st.sidebar.radio("Input Method", ["SMILES String", "Draw Molecule"])

if input_mode == "SMILES String":
    smiles = st.sidebar.text_area("SMILES", height=120)
else:
    smiles = st_ketcher("") if KETCHER_AVAILABLE else st.sidebar.text_area("SMILES", height=120)

smiles = smiles.strip()
if not smiles or not validate_smiles(smiles):
    st.info("👈 Enter a valid SMILES to start")
    st.stop()

# -------------------- INPUT DISPLAY --------------------
st.markdown("<div class='input-area'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 Molecular Structure")
    st.image(draw_molecule(smiles), use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧮 Calculated Descriptors")
    desc_df = calculate_selected_descriptors(smiles, model_features)
    st.dataframe(desc_df.T.rename(columns={0: "Value"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
st.markdown("---")
st.markdown("<div class='section-title'>🎯 Prediction</div>", unsafe_allow_html=True)

if st.button("🚀 PREDICT SGLT2 ACTIVITY", use_container_width=True):
    pred = model.predict(desc_df)[0]
    prob = model.predict_proba(desc_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        box = "active" if pred == 1 else "inactive"
        label = "🟢 ACTIVE (SGLT2 Inhibitor)" if pred == 1 else "🔴 INACTIVE"
        st.markdown(f"<div class='prediction-box {box}'>{label}</div>", unsafe_allow_html=True)

    with col2:
        st.metric("Prediction Confidence", f"{prob:.1%}")

# -------------------- FOOTER --------------------
st.markdown("""
<div class='footer'>
🧪 <strong>SGLT2 Inhibitor Prediction Tool</strong> <br>
Built with ❤️ using Streamlit, RDKit, Mordred & SHAP
</div>
""", unsafe_allow_html=True)
