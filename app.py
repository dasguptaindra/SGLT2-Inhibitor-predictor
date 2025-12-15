# =========================
# SGLT2i Predictor
# =========================

import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- RDKit --------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski

# -------- Mordred --------
from mordred import Calculator, descriptors

# -------- SHAP --------
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

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1, h2, h3 { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL & FEATURES =================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    model_features = json.load(f)


# ================= INIT MORDED =================
calc = Calculator(descriptors, ignore_3D=True)

# get full Mordred descriptor list ONCE
_tmp_mol = Chem.MolFromSmiles("CC")
MORDRED_NAMES = list(calc(_tmp_mol).keys())


# ================= STRICT DESCRIPTOR MAP =================
# JSON feature name  -> EXACT Mordred descriptor name
DESCRIPTOR_MAP = {
    "MINaaN": "MINaaN",
    "MAXaaN": "MAXaaN",
    "NaaN": "NaaN",
    "MAXsCl": "MAXsCl",

    "nN": "nN",
    "nFARing": "nFARing",
    "nFAHRing": "nFAHRing",
    "naHRing": "naHRing",

    "BCUTs-1h": "BCUTs-1h",
    "ATSC2c": "ATSC2c",
    "MATS2c": "MATS2c",
    "MDEC-33": "MDEC-33",
}


# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(280, 280)) if mol else None


def calculate_descriptors(smiles: str) -> pd.DataFrame:
    """
    Correct descriptor pipeline:
    1. Calculate ALL Mordred descriptors
    2. Calculate RDKit-only descriptors
    3. Extract ONLY model-required features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # -------- 1. Mordred descriptors --------
    mordred_result = calc(mol)

    mordred_dict = {}
    for name in MORDRED_NAMES:
        try:
            value = mordred_result[name]
            mordred_dict[name] = float(value) if value is not None else 0.0
        except:
            mordred_dict[name] = 0.0

    # -------- 2. RDKit descriptors --------
    rdkit_dict = {
        "nHBAcc_Lipinski": float(Lipinski.NumHAcceptors(mol))
    }

    # -------- 3. Build model input --------
    final_features = {}

    for feature in model_features:
        if feature in rdkit_dict:
            final_features[feature] = rdkit_dict[feature]

        elif feature in DESCRIPTOR_MAP:
            mordred_name = DESCRIPTOR_MAP[feature]
            final_features[feature] = mordred_dict.get(mordred_name, 0.0)

        else:
            final_features[feature] = 0.0

    df = pd.DataFrame([final_features])
    df = df.replace([np.inf, -np.inf, np.nan], 0.0)

    return df


# ================= HEADER =================
st.title("SGLT2i Predictor v1.0")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write(
        "**SGLT2i Predictor** predicts SGLT2 inhibitory activity of drug-like molecules "
        "using a Gradient Boosting model with SHAP-based interpretation."
    )


# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Draw Molecule")
    if KETCHER_AVAILABLE:
        smiles_drawn = st_ketcher()
    else:
        smiles_drawn = ""

with col2:
    st.subheader("🧬 SMILES Input")
    smiles = st.text_input(
        "Enter SMILES",
        value=smiles_drawn if smiles_drawn else
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    )


# ================= VALIDATION =================
if not smiles:
    st.stop()

if not validate_smiles(smiles):
    st.error("Invalid SMILES")
    st.stop()


# ================= RESULTS =================
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    desc_df = calculate_descriptors(smiles)

    pred = model.predict(desc_df)[0]
    prob = model.predict_proba(desc_df)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Query Molecule")

        if pred == 1:
            st.success("🟢 ACTIVE – SGLT2 Inhibitor")
        else:
            st.error("🔴 INACTIVE – Non-Inhibitor")

        st.metric("Confidence Score", f"{prob:.2%}")

    with col2:
        st.subheader("📈 SHAP Interpretation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(desc_df)

        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]
            base_val = explainer.expected_value[1]
        else:
            shap_val = shap_values[0]
            base_val = explainer.expected_value

        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=desc_df.iloc[0].values,
            feature_names=desc_df.columns
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()


# ================= DESCRIPTORS =================
with st.expander("🔬 Calculated Descriptors"):
    st.dataframe(desc_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ================= SIDEBAR =================
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write("""
    **Algorithm:** Gradient Boosting  
    **Descriptors:** Mordred + RDKit  
    **Key AtomTypeEState:** MINaaN, MAXaaN  
    **Interpretability:** SHAP  
    """)

    st.info(
        "💡 Try Dapagliflozin:\n"
        "CCCCCC1=CC(=C(C(=C1)C2C(C(C(O2)CO)O)O)"
        "OC3C(C(C(C(O3)CO)O)O)O)O)"
    )
