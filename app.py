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


def debug_mordred_descriptors(smiles: str):
    """Debug function to see what Mordred actually computes"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES for debug")
        return
    
    mordred_result = calc(mol)
    
    st.subheader("🔍 Mordred Debug Info")
    
    # Show all available descriptors
    mordred_dict = mordred_result.asdict()
    st.write(f"Total Mordred descriptors available: {len(mordred_dict)}")
    
    # Check your required descriptors
    st.write("**Looking for your descriptors in Mordred output:**")
    for feat in model_features:
        if feat != "nHBAcc_Lipinski":
            found = False
            for mordred_key in mordred_dict.keys():
                if feat.lower() == mordred_key.lower():
                    st.write(f"✓ Found '{feat}' as '{mordred_key}' in Mordred")
                    found = True
                    break
            if not found:
                st.write(f"✗ NOT FOUND: {feat}")
    
    # Show first few descriptors as example
    st.write("**Sample of available Mordred descriptors:**")
    sample_dict = {k: mordred_dict[k] for k in list(mordred_dict.keys())[:20]}
    st.json(sample_dict)


def calculate_descriptors(smiles: str):
    """
    Robust descriptor calculation with fallback strategies
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    # Calculate all Mordred descriptors
    mordred_result = calc(mol)
    
    # Dictionary to store descriptor values
    data = {}
    failed_descriptors = []
    successful_descriptors = []
    
    # Pre-calculate all available Mordred descriptor values
    mordred_dict = {}
    for desc_name, desc_value in mordred_result.asdict().items():
        try:
            # Convert to string and check if valid
            if desc_value is None:
                mordred_dict[desc_name] = None
                continue
                
            str_val = str(desc_value)
            if str_val.lower() == "nan" or str_val.lower() == "inf" or str_val.lower() == "-inf":
                mordred_dict[desc_name] = None
            else:
                mordred_dict[desc_name] = float(desc_value)
        except:
            mordred_dict[desc_name] = None
    
    # Calculate each required descriptor
    for feat in model_features:
        # Special case: RDKit Lipinski descriptor
        if feat == "nHBAcc_Lipinski":
            try:
                data[feat] = float(Lipinski.NumHAcceptors(mol))
                successful_descriptors.append(feat)
            except Exception as e:
                data[feat] = np.nan
                failed_descriptors.append(feat)
            continue
        
        # Mordred descriptors
        try:
            # Try exact match first
            if feat in mordred_dict:
                val = mordred_dict[feat]
                
                if val is None or np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Invalid value for {feat}")
                
                data[feat] = val
                successful_descriptors.append(feat)
                continue
            
            # Try case-insensitive match
            found_key = None
            for key in mordred_dict.keys():
                if key.lower() == feat.lower():
                    found_key = key
                    break
            
            if found_key:
                val = mordred_dict[found_key]
                if val is None or np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Invalid value for {feat} (found as {found_key})")
                
                data[feat] = val
                successful_descriptors.append(feat)
                continue
            
            # Descriptor not found at all
            raise KeyError(f"Descriptor {feat} not found in Mordred results")
                    
        except Exception as e:
            data[feat] = np.nan
            failed_descriptors.append(feat)
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Report descriptor calculation status
    success_count = len(successful_descriptors)
    total_count = len(model_features)
    
    if success_count == total_count:
        st.success(f"✅ Successfully calculated all {total_count} descriptors")
    else:
        st.warning(f"⚠️ Calculated {success_count}/{total_count} descriptors")
        if failed_descriptors:
            with st.expander("View failed descriptors"):
                for failed in failed_descriptors:
                    st.write(f"  • {failed}")
    
    # ---------- SAFE IMPUTATION ----------
    # Store original failed list for reporting
    original_failed = failed_descriptors.copy()
    
    # Calculate column medians from training data or use fallback
    # For now, we'll use a simple fallback strategy
    fallback_values = {
        'MAXaaN': 0.0, 'MINaaN': 0.0, 'nN': 0.0, 'nFARing': 0.0,
        'naHRing': 0.0, 'MAXsCl': 0.0, 'NaaN': 0.0, 'nHBAcc_Lipinski': 0.0,
        'BCUTs-1h': 0.0, 'nFAHRing': 0.0, 'ATSC2c': 0.0, 
        'MDEC-33': 0.0, 'MATS2c': 0.0
    }
    
    # Apply imputation
    for col in df.columns:
        if pd.isna(df[col].iloc[0]):
            if col in fallback_values:
                df[col] = fallback_values[col]
                if col in failed_descriptors:
                    failed_descriptors.remove(col)
            else:
                df[col] = 0.0
    
    # Ensure all values are finite
    df = df.replace([np.inf, -np.inf], 0)
    
    # Verify all descriptors are present and finite
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = fallback_values.get(feat, 0.0)
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    return df, original_failed


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

# Add debug option
debug_mode = st.checkbox("Enable debug mode (show descriptor calculation details)")

# ================= VALIDATION =================
if not smiles:
    st.info("Please draw a molecule or enter a SMILES string.")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string.")
    st.stop()

# Debug mode if enabled
if debug_mode:
    debug_mordred_descriptors(smiles)

# ================= DESCRIPTOR CALCULATION =================
with st.spinner("Calculating molecular descriptors..."):
    try:
        desc_df, failed_desc = calculate_descriptors(smiles)
        
        if len(failed_desc) == len(model_features):
            st.error("Failed to calculate all descriptors. Cannot make prediction.")
            st.stop()
            
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
        st.info("SHAP interpretation requires tree-based models. Using fallback visualization.")
        
        # Fallback: Show feature importance from model
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': desc_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Important Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

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
    
    # Show statistics
    st.write("**Descriptor Statistics:**")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric("Mean", f"{desc_df.values.mean():.4f}")
        st.metric("Std Dev", f"{desc_df.values.std():.4f}")
    with col_stats2:
        st.metric("Min", f"{desc_df.values.min():.4f}")
        st.metric("Max", f"{desc_df.values.max():.4f}")

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
    
    st.header("⚙️ Settings")
    show_raw = st.checkbox("Show raw descriptor values", value=False)
    
    if show_raw:
        st.write("**Raw descriptor values:**")
        st.json(desc_df.iloc[0].to_dict())
