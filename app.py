# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time

# Defensive imports
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Lipinski, AllChem
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

# streamlit-ketcher optional
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

# padelpy (preferred) - optional
PADELPY_AVAILABLE = False
try:
    from padelpy import padeldescriptor
    PADELPY_AVAILABLE = True
except Exception as e:
    st.info("padelpy not available: will try local PaDEL jar fallback if present.")

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# minimal CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight:700; color:#1f77b4; text-align:center}
    .sub-header {font-size:1.1rem; color:#2e86ab; text-align:center}
    .input-area {background:#f8f9fa; padding:12px; border-radius:8px}
    .prediction-box {padding:18px; border-radius:10px; text-align:center}
    .active {background:#d4edda; border:2px solid #c3e6cb; color:#155724}
    .inactive {background:#f8d7da; border:2px solid #f5c6cb; color:#721c24}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL & FEATURES --------------------
MODEL_PATH = "random_forest_model.joblib"
FEATURES_PATH = "model_features.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your model file in the app directory.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"Features file not found: {FEATURES_PATH}. Place your model_features.json in the app directory.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)

# -------------------- PaDEL helper --------------------

def run_padel_and_read(smiles: str, use_jar_path: str = None) -> pd.DataFrame | None:
    """Run padelpy.padeldescriptor and return the DataFrame (first row is molecule).

    - If padelpy is installed, this uses padeldescriptor API.
    - If padelpy is not installed but a local PaDEL-Descriptor.jar exists, padeldescriptor can still
      be invoked by passing padel_path parameter (padelpy will call java -jar).
    - Returns None on failure.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            smi_file = os.path.join(td, "input.smi")
            out_file = os.path.join(td, "output.csv")
            with open(smi_file, "w") as fh:
                fh.write(f"{smiles}\tMOL\n")

            # build kwargs for padeldescriptor
            kwargs = {
                "smiles": smi_file,
                "descriptors": True,
                "fingerprints": False,
                "output": out_file,
            }
            # if user provided jar path, pass it
            if use_jar_path:
                kwargs["padel_path"] = use_jar_path

            # call padeldescriptor (padelpy)
            padeldescriptor(**kwargs)

            if os.path.exists(out_file):
                df = pd.read_csv(out_file)
                return df
            else:
                return None
    except Exception as e:
        st.warning(f"PaDEL run failed: {e}")
        return None


def calculate_atsc2c_from_padel(smiles: str) -> (float, str):
    """Compute ATSC2c using padelpy or local jar if available. Returns (value, method).
    Method is 'padelpy', 'padel-jar', or 'failed'.
    """
    # Try padelpy default
    if PADELPY_AVAILABLE:
        try:
            df = run_padel_and_read(smiles, use_jar_path=None)
            if df is not None and "ATSC2c" in df.columns:
                val = df["ATSC2c"].iloc[0]
                if pd.isna(val):
                    return 0.0, "padelpy (nan)"
                return float(val), "padelpy"
        except Exception as e:
            st.info(f"padelpy attempt failed: {e}")

    # Try local jar fallback if present
    jar_candidates = ["PaDEL-Descriptor.jar", "paDEL/PaDEL-Descriptor.jar", "./PaDEL-Descriptor/DescriptorCalculator.jar"]
    for jar in jar_candidates:
        if os.path.exists(jar):
            df = run_padel_and_read(smiles, use_jar_path=jar)
            if df is not None and "ATSC2c" in df.columns:
                val = df["ATSC2c"].iloc[0]
                if pd.isna(val):
                    return 0.0, f"padel-jar ({jar})"
                return float(val), f"padel-jar ({jar})"

    return 0.0, "failed"

# -------------------- RDKit fallback for ATSC2c --------------------

def calculate_atsc2c_rdkit(mol: Chem.Mol) -> float:
    """A simplified RDKit-based approximation for ATSC2c (centered Broto-Moreau autocorrelation, lag 2, charge-weighted).
    This is only an approximation and should be used when PaDEL is not available.
    """
    try:
        if mol is None:
            return 0.0
        mol_h = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol_h)
        charges = []
        for atom in mol_h.GetAtoms():
            if atom.HasProp('_GasteigerCharge'):
                try:
                    c = float(atom.GetProp('_GasteigerCharge'))
                except Exception:
                    c = 0.0
            else:
                c = 0.0
            charges.append(c)

        if len(charges) < 3:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(charges) - 2):
            total += charges[i] * charges[i + 2]
            count += 1

        return (total / count) if count > 0 else 0.0
    except Exception as e:
        st.warning(f"RDKit ATSC2c fallback failed: {e}")
        return 0.0

# -------------------- Descriptor calculation function --------------------

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame | None:
    """Calculate a DataFrame with descriptors for the requested features in the same order.

    Priority of sources:
      1. PaDEL (padelpy or local jar)
      2. Mordred (if installed)
      3. RDKit simple descriptors / heuristics
      4. RDKit fallback for ATSC2c if PaDEL failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1) Try PaDEL for ATSC2c and possibly many other descriptors
    padel_values = {}
    atsc_val, atsc_method = calculate_atsc2c_from_padel(smiles)
    if atsc_method != "failed":
        padel_values['ATSC2c'] = atsc_val

    # If padel produced a larger set of descriptors, try to extract them
    if PADELPY_AVAILABLE:
        try:
            df_p = run_padel_and_read(smiles, use_jar_path=None)
            if df_p is not None:
                # convert first row to dict, drop Name column
                row = df_p.iloc[0].to_dict()
                row.pop('Name', None)
                # store everything
                for k, v in row.items():
                    padel_values[str(k)] = (0.0 if pd.isna(v) else float(v))
        except Exception:
            pass
    else:
        # Try jar fallback to get full table
        jar_candidates = ["PaDEL-Descriptor.jar", "paDEL/PaDEL-Descriptor.jar", "./PaDEL-Descriptor/DescriptorCalculator.jar"]
        for jar in jar_candidates:
            if os.path.exists(jar):
                try:
                    df_p = run_padel_and_read(smiles, use_jar_path=jar)
                    if df_p is not None:
                        row = df_p.iloc[0].to_dict()
                        row.pop('Name', None)
                        for k, v in row.items():
                            padel_values[str(k)] = (0.0 if pd.isna(v) else float(v))
                        break
                except Exception:
                    continue

    # 2) Mordred
    mordred_values = {}
    if Calculator is not None:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            res = calc(mol)
            for k, v in res.items():
                try:
                    if v is None or str(v) == 'nan':
                        mordred_values[str(k)] = 0.0
                    else:
                        mordred_values[str(k)] = float(v)
                except Exception:
                    mordred_values[str(k)] = 0.0
        except Exception as e:
            st.info(f"Mordred calculation failed: {e}")

    # 3) RDKit simple descriptors
    rdkit_values = {}
    try:
        rdkit_values['nHBAcc_Lipinski'] = float(Lipinski.NumHAcceptors(mol))
        rdkit_values['nHBDon_Lipinski'] = float(Lipinski.NumHDonors(mol))
    except Exception:
        pass

    # 4) Build feature vector in requested order
    final = {}
    for feat in features:
        # priority: padel_values > mordred_values > rdkit_values > heuristics (atom counts) > 0.0
        if feat in padel_values:
            final[feat] = padel_values.get(feat, 0.0)
        elif feat in mordred_values:
            final[feat] = mordred_values.get(feat, 0.0)
        elif feat in rdkit_values:
            final[feat] = rdkit_values.get(feat, 0.0)
        else:
            # heuristic: atom counts like nN, nO, nCl etc.
            try:
                if isinstance(feat, str) and feat.startswith('n') and len(feat) <= 4:
                    symbol = feat[1:]
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    final[feat] = float(count)
                elif feat == 'ATSC2c':
                    # if ATSC2c wasn't found earlier, compute RDKit fallback
                    val = padel_values.get('ATSC2c', None)
                    if val is None or val == 0.0:
                        final[feat] = calculate_atsc2c_rdkit(mol)
                    else:
                        final[feat] = val
                else:
                    final[feat] = 0.0
            except Exception:
                final[feat] = 0.0

    # Include meta column about ATSC2c method if requested by UI
    final['_ATSC2c_Method'] = atsc_method

    return pd.DataFrame([final], columns=[*features, '_ATSC2c_Method'])

# -------------------- Small helpers --------------------

def validate_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(400, 400))


def create_manual_waterfall(shap_val, base_value, desc_df):
    features_sorted = sorted(zip(desc_df.columns, shap_val, desc_df.iloc[0].values), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)[:10]
    features = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    actual_values = [f[2] for f in features_sorted]
    cumulative = base_value
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(features)), values)
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        label_pos = height + (0.01 if height >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width()/2., label_pos, f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    ax.set_xlabel('Features')
    ax.set_ylabel('SHAP Value')
    ax.set_title('Top 10 Feature Contributions (Waterfall Plot)')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-')
    plt.tight_layout()
    return fig

# -------------------- UI --------------------

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div style="text-align:center; font-size: 3rem;">🧪💊</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">SGLT2 Inhibitor Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Molecular Activity Using Machine Learning</div>', unsafe_allow_html=True)

st.markdown('---')

st.sidebar.header("🔬 Input Molecule")
input_mode = st.sidebar.radio("Input Method:", ["SMILES String", "Draw Molecule"]) 

if input_mode == "SMILES String":
    smiles = st.sidebar.text_area("SMILES Notation:", value="", height=100, placeholder="Enter SMILES string here...")
else:
    if KETCHER_AVAILABLE:
        smiles = st_ketcher("", key="ketcher")
    else:
        st.sidebar.warning("Ketcher not available — using SMILES input")
        smiles = st.sidebar.text_area("SMILES Notation:", value="", height=100)

smiles = (smiles or "").strip()

if not smiles:
    st.info("👈 Enter a SMILES string or draw a molecule in the sidebar to start")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string — please check your input")
    st.stop()

st.markdown('<div class="input-area">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📊 Input Molecule")
    mol_img = draw_molecule(smiles)
    if mol_img is not None:
        st.image(mol_img, caption="Molecular Structure", use_column_width=True)
    else:
        st.error("Could not generate molecule image")

with col2:
    st.subheader("🧮 Descriptor Calculation")
    with st.spinner("🔄 Calculating molecular descriptors..."):
        desc_df = calculate_selected_descriptors(smiles, model_features)

    if desc_df is None:
        st.error("❌ Descriptor calculation failed. Check RDKit/Mordred/Padel installation.")
        st.stop()

    # Show descriptors with status
    st.write("**Calculated Descriptors:**")
    desc_display = desc_df.T.rename(columns={0: 'Value'}) if desc_df.shape[1] == 1 else desc_df.T
    # remove internal method column
    if '_ATSC2c_Method' in desc_display.index:
        desc_display = desc_display.drop('_ATSC2c_Method', errors='ignore')
    # create status
    try:
        values = desc_df.iloc[0][[c for c in desc_df.columns if c != '_ATSC2c_Method']]
        status = ['⚠️' if v == 0.0 else '✅' for v in values]
        display_df = pd.DataFrame({'Value': values.values, 'Status': status}, index=values.index)
        st.dataframe(display_df, use_container_width=True)
    except Exception:
        st.dataframe(desc_df, use_container_width=True)

    # Show ATSC2c calculation method
    if '_ATSC2c_Method' in desc_df.columns:
        atsc_method = desc_df['_ATSC2c_Method'].iloc[0]
        st.info(f"ATSC2c calculation method: {atsc_method}")
        if atsc_method.startswith('padel-jar'):
            st.warning('Using local PaDEL JAR fallback — ensure Java is installed on the host')
        elif atsc_method == 'failed':
            st.warning('PaDEL not available; using RDKit fallback for ATSC2c — may be approximate')

    zero_count = (desc_df.iloc[0] == 0.0).sum()
    if zero_count > len(model_features) * 0.5:
        st.warning(f"⚠️ {zero_count} descriptors calculated as zero. This may affect prediction accuracy.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
st.markdown('---')
st.subheader('🎯 Make Prediction')
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🚀 PREDICT SGLT2 ACTIVITY")

if predict_clicked:
    with st.spinner('🤖 Making prediction...'):
        try:
            # align desc_df columns to model_features
            X = desc_df[model_features]
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.stop()

        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X)[0]
                if len(probs) == 2:
                    prob = float(probs[1])
                else:
                    try:
                        idx = list(model.classes_).index(1)
                        prob = float(probs[idx])
                    except Exception:
                        prob = None
            except Exception:
                prob = None

    st.markdown('---')
    st.subheader('📊 Prediction Results')
    c1, c2 = st.columns(2)
    with c1:
        if pred == 1:
            st.markdown('<div class="prediction-box active">', unsafe_allow_html=True)
            st.markdown('## 🟢 ACTIVE')
            st.markdown('**Predicted as SGLT2 Inhibitor**')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box inactive">', unsafe_allow_html=True)
            st.markdown('## 🔴 INACTIVE')
            st.markdown('**Not predicted as SGLT2 Inhibitor**')
            st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        if prob is not None:
            st.metric(label='**Confidence Score**', value=f"{prob:.1%}", delta='High' if prob>0.7 else 'Medium' if prob>0.5 else 'Low')
        else:
            st.info('Probability not available for this model')

    # SHAP interpretation
    if shap is not None:
        st.markdown('---')
        st.subheader('📈 Model Interpretation')
        with st.spinner('🔍 Generating SHAP explanation...'):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
                # normalize shap values to a vector for the single sample
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        shap_val = shap_values[1]
                        if len(shap_val.shape) == 2:
                            shap_val = shap_val[0]
                        base_value = expected_value[1] if hasattr(expected_value, '__len__') and len(expected_value)>1 else expected_value
                    else:
                        shap_val = shap_values[0]
                        if len(shap_val.shape) == 2:
                            shap_val = shap_val[0]
                        base_value = expected_value[0] if hasattr(expected_value, '__len__') else expected_value
                else:
                    if len(shap_values.shape) == 3:
                        shap_val = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                        base_value = expected_value[1] if hasattr(expected_value, '__len__') and len(expected_value)>1 else expected_value
                    elif len(shap_values.shape) == 2:
                        shap_val = shap_values[0]
                        base_value = expected_value
                    else:
                        shap_val = None

                if shap_val is not None:
                    shap_val = np.array(shap_val).flatten()
                    # select top features
                    top_idx = np.argsort(np.abs(shap_val))[-10:][::-1]
                    top_feats = X.columns[top_idx]
                    top_vals = shap_val[top_idx]
                    top_actuals = X.iloc[0].values[top_idx]
                    explanation = shap.Explanation(values=top_vals, base_values=base_value, data=top_actuals, feature_names=top_feats.tolist())
                    fig, ax = plt.subplots(figsize=(10,6))
                    try:
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except Exception:
                        fig = create_manual_waterfall(top_vals, base_value, X[top_feats])
                        st.pyplot(fig)
                        plt.close()
            except Exception as e:
                st.warning(f"SHAP generation failed: {e}")

# -------------------- FOOTER --------------------
st.markdown('---')
st.markdown("""
<div style='text-align:center; color:#666;'>
  <p>🧪 <strong>SGLT2 Inhibitor Prediction Tool</strong> | Built with Streamlit, RDKit, Mordred (optional), and PaDEL (padelpy or jar)</p>
</div>
""", unsafe_allow_html=True)
