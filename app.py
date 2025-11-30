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

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #2e86ab; text-align: center; margin-bottom: 2rem; }
    .drug-container { display: flex; justify-content: center; align-items: center; margin-bottom: 2rem; }
    .prediction-box { padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center; }
    .active { background-color: #d4edda; border: 2px solid #c3e6cb; color: #155724; }
    .inactive { background-color: #f8d7da; border: 2px solid #f5c6cb; color: #721c24; }
    .predict-button { width: 100%; background-color: #1f77b4; color: white; font-size: 1.2rem;
                      font-weight: bold; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; margin-top: 20px; }
    .predict-button:hover { background-color: #1668a1; transform: translateY(-2px); transition: all 0.3s ease; }
    .input-area { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 20px; }
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

# -------------------- HELPER FUNCTIONS --------------------
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame:
    """Calculate selected descriptors using Mordred and RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mordred_dict = {}
    if Calculator and descriptors:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_results = calc(mol)
            for k, v in mordred_results.items():
                try:
                    # Convert to float and handle NaN/None values
                    if v is None or str(v) == 'nan' or str(v) == 'NaN':
                        mordred_dict[str(k)] = 0.0
                    else:
                        mordred_dict[str(k)] = float(v)
                except (ValueError, TypeError):
                    mordred_dict[str(k)] = 0.0
        except Exception as e:
            st.warning(f"Mordred calculation failed: {e}")

    rdkit_dict = {}
    try:
        rdkit_dict["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))
    except Exception:
        rdkit_dict["nHBAcc_Lipinski"] = 0.0

    feature_values = {}
    for feat in features:
        if feat in mordred_dict:
            feature_values[feat] = mordred_dict[feat]
        elif feat in rdkit_dict:
            feature_values[feat] = rdkit_dict[feat]
        else:
            # Atom counts for simple features like nN, nO, nCl
            try:
                if feat.startswith('n') and len(feat) <= 4:
                    symbol = feat[1:]
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    feature_values[feat] = float(count)
                else:
                    feature_values[feat] = 0.0
            except Exception:
                feature_values[feat] = 0.0

    # Create DataFrame and ensure no NaN values remain
    desc_df = pd.DataFrame([feature_values], columns=features)
    
    # Final check and cleanup of any remaining NaN/None/inf values
    desc_df = desc_df.fillna(0.0)
    desc_df = desc_df.replace([np.inf, -np.inf], 0.0)
    
    return desc_df

def clean_dataframe_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has no NaN, None, or inf values before prediction."""
    df_clean = df.copy()
    
    # Fill NaN/None values with 0
    df_clean = df_clean.fillna(0.0)
    
    # Replace infinite values with 0
    df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
    
    # Ensure all values are numeric
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
    
    return df_clean

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(400, 400))
    return None

def create_manual_waterfall(shap_val, base_value, desc_df):
    """Manual waterfall plot if SHAP built-in fails."""
    features_sorted = sorted(
        zip(desc_df.columns, shap_val, desc_df.iloc[0].values),
        key=lambda x: abs(x[1]), reverse=True
    )[:10]

    features, values, actual_values = zip(*features_sorted)
    cumulative = base_value
    cumulative_vals = [cumulative]
    for val in values:
        cumulative += val
        cumulative_vals.append(cumulative)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(features)), values, color=['red' if v < 0 else 'blue' for v in values])

    for i, (bar, val, act) in enumerate(zip(bars, values, actual_values)):
        height = bar.get_height()
        label_pos = height + (0.01 if height >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width()/2., label_pos, f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    ax.set_xlabel('Features')
    ax.set_ylabel('SHAP Value')
    ax.set_title('Top 10 Feature Contributions (Waterfall Plot)')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# -------------------- HEADER --------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="drug-container"><div style="font-size:4rem;">🧪💊</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">SGLT2 Inhibitor Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Gradient Boosting Model</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🔬 Input Molecule")
st.sidebar.markdown("Enter your molecule structure to predict SGLT2 inhibition activity")

input_mode = st.sidebar.radio("Input Method:", ["SMILES String", "Draw Molecule"])

if input_mode == "SMILES String":
    smiles = st.sidebar.text_area(
        "SMILES Notation:", height=100, placeholder="Enter SMILES here...\nExample: C1=CC=CC=C1"
    )
else:
    if KETCHER_AVAILABLE:
        smiles = st_ketcher("", key="ketcher")
        st.sidebar.markdown("🎨 Draw your molecule")
    else:
        st.sidebar.warning("Ketcher not available. Using SMILES input.")
        smiles = st.sidebar.text_area("SMILES Notation:", height=100)

smiles = smiles.strip()

if not smiles:
    st.info("👈 Enter a SMILES string or draw a molecule in the sidebar to start")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string — please check your input")
    st.stop()

# -------------------- MAIN INPUT DISPLAY --------------------
st.markdown('<div class="input-area">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Input Molecule")
    mol_img = draw_molecule(smiles)
    if mol_img:
        st.image(mol_img, caption="Molecular Structure", use_column_width=True)
    else:
        st.error("Could not generate molecule image")

with col2:
    st.subheader("🧮 Descriptor Calculation")
    with st.spinner("🔄 Calculating molecular descriptors..."):
        desc_df = calculate_selected_descriptors(smiles, model_features)

    if desc_df is None:
        st.error("❌ Descriptor calculation failed. Check Mordred/RDKit installation.")
        st.stop()

    # Check for NaN values before displaying
    nan_count = desc_df.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"⚠️ Found {nan_count} NaN values in descriptors. These will be converted to 0 for prediction.")
        desc_df = clean_dataframe_for_prediction(desc_df)

    st.write("**Calculated Descriptors:**")
    desc_display = desc_df.T.rename(columns={0: 'Value'})
    desc_display['Status'] = ['⚠️' if x == 0.0 else '✅' for x in desc_display['Value']]
    st.dataframe(desc_display, use_container_width=True)

    zero_count = (desc_df.iloc[0] == 0.0).sum()
    if zero_count > len(model_features) * 0.5:
        st.warning(f"⚠️ {zero_count} descriptors calculated as zero. May affect accuracy.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION BUTTON --------------------
st.markdown("---")
st.subheader("🎯 Make Prediction")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_clicked = st.button("🚀 PREDICT SGLT2 ACTIVITY", use_container_width=True)

if predict_clicked:
    with st.spinner("🤖 Making prediction..."):
        try:
            # Final data cleaning before prediction
            desc_df_clean = clean_dataframe_for_prediction(desc_df)
            
            # Verify no NaN values remain
            if desc_df_clean.isna().any().any():
                st.error("❌ NaN values still present after cleaning. Cannot make prediction.")
                st.stop()
                
            pred = model.predict(desc_df_clean)[0]
            prob = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(desc_df_clean)[0]
                prob = float(probs[1]) if len(probs) == 2 else None
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.stop()

    # -------------------- DISPLAY RESULTS --------------------
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.markdown('<div class="prediction-box active">## 🟢 ACTIVE<br>Predicted as SGLT2 Inhibitor</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box inactive">## 🔴 INACTIVE<br>Not predicted as SGLT2 Inhibitor</div>', unsafe_allow_html=True)

    with col2:
        if prob is not None:
            st.metric(label="**Confidence Score**", value=f"{prob:.1%}",
                      delta="High confidence" if prob > 0.7 else "Medium confidence" if prob > 0.5 else "Low confidence")
        else:
            st.info("Probability not available for this model")

    # -------------------- SHAP INTERPRETATION --------------------
    if shap:
        st.markdown("---")
        st.subheader("📈 Model Interpretation")
        with st.spinner("🔍 Generating SHAP explanation..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(desc_df_clean)
                expected_value = explainer.expected_value

                if isinstance(shap_values, list):
                    shap_val = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                    if shap_val.ndim == 2:
                        shap_val = shap_val[0]
                    base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1 else expected_value
                else:
                    shap_val = shap_values.flatten()
                    base_value = expected_value

                # Plot SHAP waterfall
                st.subheader("Feature Contribution Analysis")
                try:
                    explanation = shap.Explanation(
                        values=shap_val[:10],
                        base_values=base_value,
                        data=desc_df_clean.iloc[0].values[:10],
                        feature_names=desc_df_clean.columns[:10].tolist()
                    )
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.waterfall(explanation, show=False)
                    st.pyplot(fig)
                    plt.close()
                except Exception:
                    st.warning("SHAP waterfall plot failed. Creating manual plot...")
                    fig = create_manual_waterfall(shap_val, base_value, desc_df_clean)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🧪 <strong>SGLT2 Inhibitor Prediction Tool</strong> | Built with Streamlit, RDKit, and Gradient Boosting
</div>
""", unsafe_allow_html=True)
