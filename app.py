# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import subprocess
import tempfile
import time

# Defensive imports
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
    # We'll continue but Mordred features may be unavailable
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

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 2rem;
    }
    .drug-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .active {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .inactive {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .predict-button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 20px;
    }
    .predict-button:hover {
        background-color: #1668a1;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .input-area {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL & FEATURES --------------------
MODEL_PATH = "random_forest_model.joblib"
FEATURES_PATH = "model_features.json"
PADEL_PATH = "PaDEL-Descriptor"  # Path to PaDEL-Descriptor directory

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your model file in the app directory.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"Features file not found: {FEATURES_PATH}. Place your model_features.json in the app directory.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)

# -------------------- PADEL DESCRIPTOR CALCULATION --------------------
def calculate_atsc2c_with_padel(smiles: str) -> float:
    """
    Calculate ATSC2c descriptor using PaDEL-Descriptor
    Returns the ATSC2c value or 0.0 if calculation fails
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as smi_file:
            smi_file.write(f"{smiles}\tTestMol\n")
            smi_path = smi_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Determine the PaDEL executable path
            if os.path.exists(os.path.join(PADEL_PATH, "DescriptorCalculator.jar")):
                padel_jar = os.path.join(PADEL_PATH, "DescriptorCalculator.jar")
            elif os.path.exists("DescriptorCalculator.jar"):
                padel_jar = "DescriptorCalculator.jar"
            else:
                st.warning("PaDEL-Descriptor not found. Using fallback calculation.")
                return 0.0
            
            # Run PaDEL-Descriptor
            cmd = [
                'java', '-jar', padel_jar,
                '-removesalt',  # Remove salt
                '-standardizenitro',  # Standardize nitro groups
                '-fingerprints',  # Calculate fingerprints
                '-descriptortypes', 'Descriptors.xml',  # Use all descriptors
                '-dir', os.path.dirname(smi_path),
                '-file', output_path,
                '-2d'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                st.warning(f"PaDEL execution failed: {result.stderr}")
                return 0.0
            
            # Read results
            if os.path.exists(output_path):
                df = pd.read_csv(output_path)
                if 'ATSC2c' in df.columns:
                    atsc2c_value = df['ATSC2c'].iloc[0]
                    # Handle NaN or infinite values
                    if pd.isna(atsc2c_value) or np.isinf(atsc2c_value):
                        return 0.0
                    return float(atsc2c_value)
                else:
                    st.warning("ATSC2c not found in PaDEL output")
                    return 0.0
            else:
                st.warning("PaDEL output file not generated")
                return 0.0
                
        except subprocess.TimeoutExpired:
            st.warning("PaDEL calculation timed out")
            return 0.0
        except Exception as e:
            st.warning(f"PaDEL calculation error: {e}")
            return 0.0
        finally:
            # Clean up temporary files
            try:
                os.unlink(smi_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
                
    except Exception as e:
        st.warning(f"PaDEL setup error: {e}")
        return 0.0

# -------------------- HELPERS --------------------
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
    if Calculator is not None and descriptors is not None:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_results = calc(mol)
            # convert
            for k, v in mordred_results.items():
                try:
                    # Handle None and NaN values properly
                    if v is None or str(v) == 'nan':
                        mordred_dict[str(k)] = 0.0
                    else:
                        mordred_dict[str(k)] = float(v)
                except Exception:
                    mordred_dict[str(k)] = 0.0
        except Exception as e:
            st.warning(f"Mordred calculation failed: {e}")
            mordred_dict = {}

    # RDKit fallback for some features
    rdkit_dict = {}
    try:
        rdkit_dict["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))
    except Exception:
        pass

    # Calculate ATSC2c using PaDEL
    atsc2c_value = 0.0
    if 'ATSC2c' in features:
        with st.spinner("🔄 Calculating ATSC2c with PaDEL..."):
            atsc2c_value = calculate_atsc2c_with_padel(smiles)

    feature_values = {}
    for feat in features:
        if feat == 'ATSC2c':
            feature_values[feat] = atsc2c_value
        elif feat in mordred_dict:
            feature_values[feat] = mordred_dict.get(feat, 0.0)
        elif feat in rdkit_dict:
            feature_values[feat] = rdkit_dict.get(feat, 0.0)
        else:
            # simple heuristic for atom counts like nN, nO, nCl etc.
            try:
                if isinstance(feat, str) and feat.startswith('n') and len(feat) <= 4:
                    symbol = feat[1:]
                    # handle multi-letter elements like Cl, Br
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    feature_values[feat] = float(count)
                else:
                    # For other descriptors that Mordred couldn't calculate, set to 0.0
                    feature_values[feat] = 0.0
            except Exception:
                feature_values[feat] = 0.0

    return pd.DataFrame([feature_values], columns=features)

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(400, 400))

def create_manual_waterfall(shap_val, base_value, desc_df):
    """Create a manual waterfall plot when SHAP's built-in fails"""
    # Sort features by contribution magnitude and take top 10
    features_sorted = sorted(zip(desc_df.columns, shap_val, desc_df.iloc[0].values), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)[:10]  # Limit to top 10
    
    features = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    actual_values = [f[2] for f in features_sorted]
    
    # Calculate cumulative values
    cumulative = base_value
    cumulative_vals = [cumulative]
    for val in values:
        cumulative += val
        cumulative_vals.append(cumulative)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars = ax.bar(range(len(features)), values, color=['red' if v < 0 else 'blue' for v in values])
    
    # Add value labels on bars
    for i, (bar, val, actual_val) in enumerate(zip(bars, values, actual_values)):
        height = bar.get_height()
        label_pos = height + (0.01 if height >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width()/2., label_pos,
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('SHAP Value')
    ax.set_title('Top 10 Feature Contributions (Waterfall Plot)')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# -------------------- HEADER SECTION --------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="drug-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 4rem; text-align: center;">🧪💊</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">SGLT2 Inhibitor Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Molecular Activity Using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🔬 Input Molecule")
st.sidebar.markdown("Enter your molecule structure to predict SGLT2 inhibition activity")

input_mode = st.sidebar.radio("Input Method:", ["SMILES String", "Draw Molecule"])

if input_mode == "SMILES String":
    smiles = st.sidebar.text_area(
        "SMILES Notation:",
        value="",  # No default value
        height=100,
        placeholder="Enter SMILES string here...\nExample: C1=CC=CC=C1 for benzene"
    )
    st.sidebar.markdown("💡 **Tip:** Use common SMILES notation for organic molecules")
else:
    if KETCHER_AVAILABLE:
        smiles = st_ketcher("", key="ketcher")  # No default value
        st.sidebar.markdown("🎨 **Draw** your molecule in the editor above")
    else:
        st.sidebar.warning("Ketcher not available. Using SMILES input.")
        smiles = st.sidebar.text_area(
            "SMILES Notation:",
            value="",  # No default value
            height=100,
            placeholder="Enter SMILES string here..."
        )

smiles = smiles.strip()

# -------------------- MAIN PREDICTION AREA --------------------
if not smiles:
    st.info("👈 Enter a SMILES string or draw a molecule in the sidebar to start")
    st.stop()

# Validate SMILES
if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string — please check your input")
    st.stop()

# Display molecule and input area
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
        st.error("❌ Descriptor calculation failed. Check Mordred/RDKit installation.")
        st.stop()

    # Show descriptors with status
    st.write("**Calculated Descriptors:**")
    desc_display = desc_df.T.rename(columns={0: 'Value'})
    desc_display['Status'] = ['⚠️' if x == 0.0 else '✅' for x in desc_display['Value']]
    st.dataframe(desc_display, use_container_width=True)

    # Highlight ATSC2c calculation method
    if 'ATSC2c' in desc_df.columns:
        atsc2c_value = desc_df['ATSC2c'].iloc[0]
        st.info(f"**ATSC2c Calculation:** {atsc2c_value:.6f} (calculated using PaDEL)")

    # Show warning if many descriptors are zero
    zero_count = (desc_df.iloc[0] == 0.0).sum()
    if zero_count > len(model_features) * 0.5:
        st.warning(f"⚠️ {zero_count} descriptors calculated as zero. This may affect prediction accuracy.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION BUTTON --------------------
st.markdown("---")
st.subheader("🎯 Make Prediction")

# Single predict button - centered and prominent
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button(
        "🚀 PREDICT SGLT2 ACTIVITY", 
        key="predict_button",
        use_container_width=True
    )

# Only proceed if predict button is clicked
if predict_clicked:
    with st.spinner("🤖 Making prediction..."):
        try:
            pred = model.predict(desc_df)[0]
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.stop()

        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(desc_df)[0]
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

    # Display prediction result
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.markdown('<div class="prediction-box active">', unsafe_allow_html=True)
            st.markdown("## 🟢 ACTIVE")
            st.markdown("**Predicted as SGLT2 Inhibitor**")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box inactive">', unsafe_allow_html=True)
            st.markdown("## 🔴 INACTIVE")
            st.markdown("**Not predicted as SGLT2 Inhibitor**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if prob is not None:
            st.metric(
                label="**Confidence Score**",
                value=f"{prob:.1%}",
                delta="High confidence" if prob > 0.7 else "Medium confidence" if prob > 0.5 else "Low confidence"
            )
        else:
            st.info("Probability not available for this model")

    # -------------------- SHAP INTERPRETATION --------------------
    if shap is not None:
        st.markdown("---")
        st.subheader("📈 Model Interpretation")
        
        with st.spinner("🔍 Generating SHAP explanation..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(desc_df)
                expected_value = explainer.expected_value

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        shap_val = shap_values[1]
                        if len(shap_val.shape) == 2:
                            shap_val = shap_val[0]
                        base_value = expected_value[1] if hasattr(expected_value, '__len__') and len(expected_value) > 1 else expected_value
                    else:
                        shap_val = shap_values[0]
                        if len(shap_val.shape) == 2:
                            shap_val = shap_val[0]
                        base_value = expected_value[0] if hasattr(expected_value, '__len__') else expected_value
                else:
                    if len(shap_values.shape) == 3:
                        shap_val = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                        base_value = expected_value[1] if hasattr(expected_value, '__len__') and len(expected_value) > 1 else expected_value
                    elif len(shap_values.shape) == 2:
                        shap_val = shap_values[0]
                        base_value = expected_value
                    else:
                        st.error(f"Unexpected SHAP values shape: {shap_values.shape}")
                        shap_val = None

                if shap_val is not None:
                    shap_val = np.array(shap_val).flatten()

                    # Create waterfall plot
                    st.subheader("Feature Contribution Analysis")
                    try:
                        # Filter significant features
                        significant_mask = np.abs(shap_val) > 0.001
                        significant_indices = np.where(significant_mask)[0]
                        
                        if len(significant_indices) > 0:
                            top_sig_indices = significant_indices[np.argsort(np.abs(shap_val[significant_indices]))[-10:][::-1]]
                            top_shap_vals = shap_val[top_sig_indices]
                            top_features = desc_df.columns[top_sig_indices]
                            top_actual_vals = desc_df.iloc[0].values[top_sig_indices]
                        else:
                            top_indices = np.argsort(np.abs(shap_val))[-10:][::-1]
                            top_shap_vals = shap_val[top_indices]
                            top_features = desc_df.columns[top_indices]
                            top_actual_vals = desc_df.iloc[0].values[top_indices]
                        
                        explanation = shap.Explanation(
                            values=top_shap_vals,
                            base_values=base_value,
                            data=top_actual_vals,
                            feature_names=top_features.tolist()
                        )
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        plt.title("Top 10 Feature Contributions to Prediction", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e:
                        st.warning(f"Waterfall plot failed: {e}")
                        st.info("Creating alternative visualization...")
                        fig = create_manual_waterfall(shap_val, base_value, desc_df)
                        st.pyplot(fig)
                        plt.close()

            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧪 <strong>SGLT2 Inhibitor Prediction Tool</strong> | Built with Streamlit, RDKit, and Machine Learning</p>
    <p>ATSC2c descriptor calculated using PaDEL-Descriptor</p>
</div>
""", unsafe_allow_html=True)