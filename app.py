# app.py

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Main container styling */
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }
    
    /* Cards */
    .card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #667eea; }
    
    /* Headers */
    .main-header { 
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 1rem;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 5px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #4299e1 0%, #667eea 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING --------------------
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found: {FEATURES_PATH}")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        model_features = json.load(f)
    return model, model_features

model, model_features = load_model()

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
            try:
                if feat.startswith('n') and len(feat) <= 4:
                    symbol = feat[1:]
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    feature_values[feat] = float(count)
                else:
                    feature_values[feat] = 0.0
            except Exception:
                feature_values[feat] = 0.0

    desc_df = pd.DataFrame([feature_values], columns=features)
    desc_df = desc_df.fillna(0.0)
    desc_df = desc_df.replace([np.inf, -np.inf], 0.0)
    
    return desc_df

def clean_dataframe_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has no NaN, None, or inf values before prediction."""
    df_clean = df.copy()
    df_clean = df_clean.fillna(0.0)
    df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
    
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
    
    return df_clean

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(350, 350))
    return None

def create_gauge_chart(value, title, color_scheme="RdYlGn"):
    """Create a gauge chart for probability display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 30], 'color': '#f56565'},
                {'range': [30, 70], 'color': '#ed8936'},
                {'range': [70, 100], 'color': '#48bb78'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        font={'size': 12}
    )
    return fig

def create_shap_waterfall_plot(shap_values, expected_value, feature_names, max_display=10):
    """Create a SHAP waterfall plot."""
    # Get sorted indices by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1][:max_display]
    sorted_shap = shap_values[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    # Calculate cumulative values
    cum_values = np.cumsum(np.insert(sorted_shap, 0, expected_value))
    
    # Colors
    colors = []
    for val in sorted_shap:
        if val >= 0:
            colors.append('#48bb78')  # Green
        else:
            colors.append('#f56565')  # Red
    
    fig = go.Figure()
    
    # Add bars for SHAP values
    fig.add_trace(go.Bar(
        y=sorted_features,
        x=sorted_shap,
        orientation='h',
        marker_color=colors,
        name='Feature Contribution',
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
    ))
    
    # Add expected value line
    fig.add_shape(
        type="line",
        x0=expected_value, x1=expected_value,
        y0=-0.5, y1=len(sorted_features)-0.5,
        line=dict(color="black", width=2, dash="dash")
    )
    
    fig.update_layout(
        title=f"SHAP Waterfall Plot (Top {max_display} Features)",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Features",
        height=400,
        template="plotly_white",
        showlegend=False,
        bargap=0.2,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    return fig

# -------------------- HEADER --------------------
st.markdown('<div class="main-header">SGLT2 Inhibitor Predictor</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #718096; font-size: 1rem; margin-bottom: 20px;">Predict SGLT2 inhibition activity using machine learning</p>', unsafe_allow_html=True)

# Decorative separator
st.markdown("""
<div style="height: 2px; background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); margin: 10px 0 20px 0;"></div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR INPUT --------------------
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### **Input Molecule**')
    st.markdown('Enter your molecule structure to predict SGLT2 inhibition activity')
    
    # Input mode selector
    input_mode = st.radio(
        "**Select Input Method:**",
        ["SMILES String", "Draw Molecule"],
        index=0
    )
    
    if input_mode == "SMILES String":
        smiles = st.text_area(
            "**SMILES Notation:**",
            height=100,
            placeholder="Enter SMILES here...\n\nExamples:\nAspirin: CC(=O)OC1=CC=CC=C1C(=O)O\nCaffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            help="Enter the SMILES string of your molecule"
        )
    else:
        if KETCHER_AVAILABLE:
            st.markdown("**Draw your molecule below:**")
            smiles = st_ketcher("", key="ketcher")
        else:
            st.warning("Ketcher not available")
            st.info("Please install streamlit-ketcher or use SMILES input")
            smiles = st.text_area("**SMILES Notation:**", height=100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Information card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### **About This Tool**')
    st.markdown("""
    This tool predicts **SGLT2 inhibition activity** using:
    - Gradient Boosting Model
    - Molecular Descriptors (Mordred & RDKit)
    - SHAP Explanations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

smiles = smiles.strip() if smiles else ""

# -------------------- MAIN CONTENT --------------------
if not smiles:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px;">
            <h2 style="color: #4a5568;">Welcome to SGLT2 Predictor</h2>
            <p style="color: #718096;">
                Start by entering a SMILES string or drawing a molecule in the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not validate_smiles(smiles):
    st.error("""
    ### **Invalid SMILES String**
    Please check your input and try again.
    """)
    st.stop()

# -------------------- MOLECULE VISUALIZATION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Molecule Visualization</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    mol_img = draw_molecule(smiles)
    if mol_img:
        st.image(mol_img, caption="Molecular Structure", use_column_width=True)
    else:
        st.error("Could not generate molecule image")

with col2:
    st.markdown("### **Descriptor Calculation**")
    
    # Simple progress indicator
    with st.spinner("Calculating descriptors..."):
        desc_df = calculate_selected_descriptors(smiles, model_features)
    
    if desc_df is None:
        st.error("Descriptor calculation failed")
        st.stop()
    
    # Clean data
    desc_df = clean_dataframe_for_prediction(desc_df)
    
    # Display compact descriptor metrics
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        non_zero = (desc_df.iloc[0] != 0.0).sum()
        st.metric("Valid Descriptors", f"{non_zero}")
    
    with col_stat2:
        zero_count = (desc_df.iloc[0] == 0.0).sum()
        st.metric("Zero Values", f"{zero_count}")
    
    # Descriptor preview
    if st.checkbox("Show descriptor values", value=False):
        st.dataframe(desc_df.T.rename(columns={0: 'Value'}), use_container_width=True, height=200)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION SECTION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Make Prediction</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Ready to Predict?
    Click the button below to analyze your molecule's SGLT2 inhibition potential.
    """)

with col2:
    predict_button = st.button("**RUN PREDICTION**", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing molecule..."):
        desc_df_clean = clean_dataframe_for_prediction(desc_df)
        
        if desc_df_clean.isna().any().any():
            st.error("Data cleaning failed. Cannot make prediction.")
            st.stop()
        
        pred = model.predict(desc_df_clean)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(desc_df_clean)[0]
            prob = float(probs[1]) if len(probs) == 2 else None
    
    # -------------------- RESULTS DISPLAY --------------------
    st.markdown("## **Prediction Results**")
    
    # Result cards
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if pred == 1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
                <h2 style="margin: 10px 0;">ACTIVE</h2>
                <p>SGLT2 Inhibitor</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);">
                <h2 style="margin: 10px 0;">INACTIVE</h2>
                <p>Not an SGLT2 Inhibitor</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if prob is not None:
            st.plotly_chart(create_gauge_chart(prob, "Confidence Score"), use_container_width=True)
    
    # -------------------- SHAP ANALYSIS --------------------
    if shap:
        st.markdown('<div class="section-header">Model Interpretation</div>', unsafe_allow_html=True)
        
        with st.spinner("Generating feature explanations..."):
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
                
                # Create waterfall plot
                st.plotly_chart(
                    create_shap_waterfall_plot(shap_val, base_value, desc_df_clean.columns, max_display=12),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
    
    # -------------------- DOWNLOAD RESULTS --------------------
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        results = {
            "smiles": smiles,
            "prediction": "Active" if pred == 1 else "Inactive",
            "probability": prob if prob else "N/A",
            "descriptors": desc_df_clean.iloc[0].to_dict()
        }
        
        import io
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="**Download Results (JSON)**",
            data=json_str,
            file_name=f"sglt2_prediction_{smiles[:10]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        if st.button("**Analyze Another Molecule**", use_container_width=True):
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div style="text-align: center; color: #718096; padding: 20px; margin-top: 30px;">
    <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 2px; width: 80px; margin: 0 auto 10px;"></div>
    <p>SGLT2 Inhibitor Prediction Tool</p>
    <p>Built with Streamlit • RDKit • Mordred • SHAP</p>
</div>
""", unsafe_allow_html=True)
