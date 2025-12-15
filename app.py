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
.stMetric { padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    MODEL_PATH = "gradient_boosting_model_fixed.joblib"
    FEATURES_PATH = "model_features.json"
    
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        model_features = json.load(f)
    
    # Initialize Mordred calculator
    calc = Calculator(descriptors, ignore_3D=True)
    
    return model, model_features, calc

try:
    model, model_features, calc = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or smiles.strip() == "":
        return False
    return Chem.MolFromSmiles(smiles) is not None

def draw_molecule(smiles):
    """Draw molecule from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None

def calculate_descriptors(smiles):
    """Calculate all Mordred descriptors and extract the ones needed by the model"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Calculate all Mordred descriptors
    try:
        mordred_result = calc(mol)
    except Exception as e:
        raise ValueError(f"Error calculating Mordred descriptors: {str(e)}")
    
    # Get all available descriptor names from Mordred
    all_descriptors = list(mordred_result.keys())
    
    # Prepare data dictionary for model features
    data = {}
    
    # Feature mapping between model feature names and Mordred descriptors
    feature_mapping = {
        'MAXaaN': ['MAXaaN', 'Eta_max_aaN', 'max_aaN', 'max_eta_aaN', 'maxaasn'],
        'MINaaN': ['MINaaN', 'Eta_min_aaN', 'min_aaN', 'min_eta_aaN', 'minaasn'],
        'nN': ['nN', 'N', 'nNCount', 'num_nitrogen', 'nN'],
        'nFARing': ['nFARing', 'nFusedAromaticRings', 'nFAR', 'num_fused_aromatic_rings'],
        'naHRing': ['naHRing', 'nAliphaticHeterocycles', 'naHR', 'num_aliphatic_heterocycles'],
        'MAXsCl': ['MAXsCl', 'Eta_max_sCl', 'max_sCl', 'max_eta_sCl', 'maxscl'],
        'NaaN': ['NaaN', 'Eta_count_aaN', 'n_aaN', 'count_aaN', 'n_eta_aaN', 'naasn'],
        'nHBAcc_Lipinski': None,  # Handled separately
        'BCUTs-1h': ['BCUTs-1h', 'BCUTS-1h', 'bcut_s_1h', 'BCUT_s_1h'],
        'nFAHRing': ['nFAHRing', 'nFusedAromaticHeterocycles', 'nFAHR', 'num_fused_aromatic_heterocycles'],
        'ATSC2c': ['ATSC2c', 'ATS2c', 'ATSC_2c', 'ATS_2c'],
        'MDEC-33': ['MDEC-33', 'MDEC33', 'mdec_33', 'mdec33'],
        'MATS2c': ['MATS2c', 'MATS2C', 'mats_2c', 'MATS_2c']
    }
    
    # Helper function to extract value from Mordred result
    def extract_mordred_value(descriptor_list):
        for desc_name in descriptor_list:
            for avail_desc in all_descriptors:
                # Case-insensitive comparison
                if desc_name.lower() == avail_desc.lower():
                    try:
                        value = mordred_result[avail_desc]
                        # Handle special types
                        if hasattr(value, 'asdict'):
                            # For atom-type E-state descriptors
                            if 'max' in desc_name.lower():
                                return float(value.asdict().get('max', 0))
                            elif 'min' in desc_name.lower():
                                return float(value.asdict().get('min', 0))
                            else:
                                return float(value)
                        elif isinstance(value, (list, tuple)):
                            # For descriptors that return arrays
                            if len(value) > 0:
                                if 'max' in desc_name.lower():
                                    return float(max(value))
                                elif 'min' in desc_name.lower():
                                    return float(min(value))
                                else:
                                    return float(value[0])
                            else:
                                return 0.0
                        else:
                            return float(value)
                    except Exception as e:
                        st.warning(f"Could not extract value for {desc_name}: {str(e)}")
                        continue
        return 0.0
    
    # Calculate each feature
    for feature in model_features:
        if feature == 'nHBAcc_Lipinski':
            # Calculate Lipinski descriptor
            try:
                data[feature] = float(Lipinski.NumHAcceptors(mol))
            except:
                data[feature] = 0.0
        elif feature in feature_mapping:
            # Get value from Mordred using the mapping
            mapping = feature_mapping[feature]
            if mapping:
                data[feature] = extract_mordred_value(mapping)
            else:
                data[feature] = 0.0
        else:
            # Try to find the descriptor directly
            data[feature] = extract_mordred_value([feature])
    
    # Create DataFrame and handle infinities/NaN
    df = pd.DataFrame([data])
    
    # Replace infinities and NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Ensure all features are present and in correct order
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    return df

def create_shap_plot(desc_df, model):
    """Create SHAP waterfall plot"""
    try:
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
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=desc_df.iloc[0].values,
            feature_names=desc_df.columns
        )
        
        # Plot top 10 features
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title("SHAP Feature Contributions", fontsize=14, pad=20)
        plt.tight_layout()
        
        return fig, shap_val, base_val
    except Exception as e:
        st.warning(f"SHAP visualization error: {str(e)}")
        return None, None, None

# ================= HEADER =================
st.title("💊 SGLT2i Predictor v1.0")
st.markdown("Predict SGLT2 inhibitory activity of small molecules")

with st.expander("ℹ️ About this tool", expanded=False):
    st.write("""
    **SGLT2i Predictor** is a machine learning-based tool that predicts the Sodium-Glucose Cotransporter 2 (SGLT2) 
    inhibitory activity of small molecules. 
    
    **Features:**
    - Predicts whether a molecule is an SGLT2 inhibitor
    - Provides confidence scores
    - SHAP-based interpretability shows feature contributions
    - Calculates 13 key molecular descriptors
    
    **Model Information:**
    - Algorithm: Gradient Boosting Classifier
    - Features: 13 molecular descriptors
    - Training: Validated on known SGLT2 inhibitors
    
    **Note:** This tool is for research purposes only.
    """)

# ================= INPUT SECTION =================
st.markdown("---")
st.subheader("🔬 Input Molecule")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### ✏️ Draw Molecule")
    if KETCHER_AVAILABLE:
        smiles_drawn = st_ketcher()
        if smiles_drawn and smiles_drawn != "":
            st.success(f"Drawn molecule loaded: {smiles_drawn[:50]}...")
    else:
        st.info("Ketcher not available. Please enter SMILES manually.")
        smiles_drawn = ""

with col2:
    st.markdown("#### 🧬 SMILES Input")
    
    # Example molecules
    example_molecules = {
        "Select example...": "",
        "Dapagliflozin (SGLT2 inhibitor)": "CCCCCC1=CC(=C(C(=C1)C2C(C(C(O2)CO)O)O)OC3C(C(C(C(O3)CO)O)O)O)O)OC4C(C(C(C(O4)CO)O)O)O",
        "Aspirin (non-inhibitor)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Canagliflozin (SGLT2 inhibitor)": "CC1=CC(=C(C=C1OC2C(C(C(O2)COC3=CC=C(C=C3)F)O)O)F)C4=CC(=C(C(=C4)Cl)O)Cl",
        "Glucose (control)": "C(C1C(C(C(C(O1)O)O)O)O)O"
    }
    
    example_choice = st.selectbox("Load example molecule:", list(example_molecules.keys()))
    
    if example_choice != "Select example...":
        default_smiles = example_molecules[example_choice]
    else:
        default_smiles = smiles_drawn if smiles_drawn else "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    smiles = st.text_area(
        "Enter SMILES string:",
        value=default_smiles,
        height=100,
        help="Enter a valid SMILES string or use the example molecules"
    )

# ================= VALIDATION =================
if not smiles or smiles.strip() == "":
    st.info("👈 Please draw a molecule or enter a SMILES string.")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string. Please check the format and try again.")
    st.stop()

# Display molecule
st.markdown("#### 📐 Molecule Visualization")
mol_img = draw_molecule(smiles)
if mol_img:
    st.image(mol_img, caption="Input Molecule", width=300)
else:
    st.warning("Could not visualize molecule")

# ================= CALCULATION & PREDICTION =================
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    # Show loading spinner
    with st.spinner("Calculating molecular descriptors and making prediction..."):
        # Calculate descriptors
        desc_df = calculate_descriptors(smiles)
        
        # Make prediction
        pred = model.predict(desc_df)[0]
        prob = model.predict_proba(desc_df)[0][1]
        confidence = prob if pred == 1 else (1 - prob)
    
    # Display results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Prediction")
        
        # Display prediction with color
        if pred == 1:
            st.success(f"""
            ### 🟢 ACTIVE - SGLT2 Inhibitor
            **Probability:** {prob:.2%}
            """)
        else:
            st.error(f"""
            ### 🔴 INACTIVE - Non-Inhibitor
            **Probability:** {(1-prob):.2%}
            """)
        
        # Confidence gauge
        st.markdown("#### 📈 Confidence")
        progress_value = confidence
        st.progress(float(progress_value))
        st.caption(f"Confidence: {confidence:.2%}")
        
        # Additional metrics
        st.markdown("#### 📋 Quick Stats")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Descriptor Count", len(model_features))
        with col_stat2:
            st.metric("Active Probability", f"{prob:.2%}")
    
    with col2:
        st.markdown("#### 🔍 SHAP Interpretation")
        
        # Create SHAP plot
        fig, shap_val, base_val = create_shap_plot(desc_df, model)
        
        if fig:
            st.pyplot(fig)
            plt.close()
            
            # SHAP values table
            with st.expander("📋 Detailed SHAP Values", expanded=False):
                if shap_val is not None:
                    shap_df = pd.DataFrame({
                        'Feature': desc_df.columns,
                        'SHAP Value': shap_val,
                        'Feature Value': desc_df.iloc[0].values,
                        'Absolute Impact': np.abs(shap_val)
                    }).sort_values('Absolute Impact', ascending=False)
                    
                    st.dataframe(
                        shap_df.head(15),
                        use_container_width=True,
                        column_config={
                            "Feature": st.column_config.TextColumn("Feature"),
                            "SHAP Value": st.column_config.NumberColumn(
                                "SHAP Value",
                                format="%.4f"
                            ),
                            "Feature Value": st.column_config.NumberColumn(
                                "Feature Value",
                                format="%.4f"
                            ),
                            "Absolute Impact": st.column_config.NumberColumn(
                                "Absolute Impact",
                                format="%.4f"
                            )
                        }
                    )
        else:
            st.info("SHAP visualization not available for this prediction.")

except Exception as e:
    st.error(f"❌ Error during prediction: {str(e)}")
    st.info("""
    **Possible issues:**
    1. The molecule may be too complex
    2. Descriptor calculation failed
    3. Model compatibility issue
    
    Try a simpler molecule or check the SMILES format.
    """)
    st.stop()

# ================= DESCRIPTOR DETAILS =================
with st.expander("🔬 Calculated Descriptors", expanded=False):
    st.markdown(f"**Model uses {len(model_features)} features:**")
    
    # Display descriptors in a nice table
    desc_table = pd.DataFrame({
        'Descriptor': desc_df.columns,
        'Value': desc_df.iloc[0].values,
        'Type': ['Atom-type E-state' if 'aaN' in feat or 'sCl' in feat else 
                'Count' if feat.startswith('n') else 
                'BCUT' if 'BCUT' in feat else 
                'Autocorrelation' if 'ATS' in feat or 'MATS' in feat else 
                'Topological' if 'MDEC' in feat else 
                'Lipinski' if 'Lipinski' in feat else 'Other'
                for feat in desc_df.columns]
    })
    
    st.dataframe(
        desc_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Descriptor": st.column_config.TextColumn("Descriptor"),
            "Value": st.column_config.NumberColumn("Value", format="%.6f"),
            "Type": st.column_config.TextColumn("Type")
        }
    )
    
    # Statistics
    st.markdown("#### 📊 Descriptor Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Mean", f"{desc_df.values.mean():.4f}")
    with stat_col2:
        st.metric("Std Dev", f"{desc_df.values.std():.4f}")
    with stat_col3:
        st.metric("Min", f"{desc_df.values.min():.4f}")
    with stat_col4:
        st.metric("Max", f"{desc_df.values.max():.4f}")

# ================= DEBUG INFORMATION =================
if st.session_state.get('debug_mode', False):
    with st.expander("🐛 Debug Information", expanded=False):
        st.write("**Model Features:**", model_features)
        st.write("**Calculated Features:**", list(desc_df.columns))
        
        # Check for mismatches
        missing = set(model_features) - set(desc_df.columns)
        extra = set(desc_df.columns) - set(model_features)
        
        if missing:
            st.warning(f"Missing features: {list(missing)}")
        if extra:
            st.info(f"Extra features calculated: {list(extra)}")
        
        st.write("**Raw descriptor values:**")
        st.json(desc_df.iloc[0].to_dict())

# ================= SIDEBAR =================
with st.sidebar:
    st.title("ℹ️ About")
    
    st.markdown("""
    ### Model Features
    
    The model uses 13 key molecular descriptors:
    
    1. **MAXaaN** - Maximum aaN atom-type E-state
    2. **MINaaN** - Minimum aaN atom-type E-state
    3. **nN** - Number of Nitrogen atoms
    4. **nFARing** - Number of fused aromatic rings
    5. **naHRing** - Number of aliphatic heterocycles
    6. **MAXsCl** - Maximum sCl atom-type E-state
    7. **NaaN** - Number of aaN atoms
    8. **nHBAcc_Lipinski** - Hydrogen bond acceptors (Lipinski)
    9. **BCUTs-1h** - BCUT descriptor
    10. **nFAHRing** - Number of fused aromatic heterocycles
    11. **ATSC2c** - Autocorrelation descriptor
    12. **MDEC-33** - Molecular distance edge descriptor
    13. **MATS2c** - Moran autocorrelation descriptor
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Example Molecules")
    st.info("""
    Try these examples:
    - **Dapagliflozin**: SGLT2 inhibitor
    - **Aspirin**: Non-inhibitor control
    - **Canagliflozin**: SGLT2 inhibitor
    - **Glucose**: Natural substrate
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox("Enable Debug Mode", False)
    st.session_state['debug_mode'] = debug_mode
    
    if debug_mode:
        st.warning("Debug mode enabled. Additional information will be shown.")
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Disclaimer")
    st.caption("""
    This tool is for **research purposes only**.
    Not for clinical decision-making.
    Always verify predictions with experimental data.
    """)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>💊 <b>SGLT2i Prediction Tool</b> | Built with Streamlit, RDKit, and Mordred</p>
    <p><small>For research use only. Predictions should be validated experimentally.</small></p>
</div>
""", unsafe_allow_html=True)
