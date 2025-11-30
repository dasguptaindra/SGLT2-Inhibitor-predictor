# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import pickle

# Configure matplotlib to avoid GUI issues
plt.switch_backend('Agg')

# Defensive imports with better error handling
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Lipinski, Descriptors
    RDKIT_AVAILABLE = True
except Exception as e:
    st.error(f"RDKit import failed: {e}")
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except Exception as e:
    st.warning(f"Mordred import warning: {e}")
    MORDRED_AVAILABLE = False
    Calculator = None
    descriptors = None

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    st.warning(f"SHAP import warning: {e}")
    SHAP_AVAILABLE = False
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
        font-size: 1.2rem;
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
        transition: all 0.3s ease;
    }
    .predict-button:hover {
        background-color: #1668a1;
        transform: translateY(-2px);
    }
    .input-area {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    .feature-positive {
        color: #28a745;
        font-weight: bold;
    }
    .feature-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL & FEATURES LOADING WITH FALLBACKS --------------------
@st.cache_resource
def load_model_with_fallback():
    """Load the trained model with multiple fallback strategies"""
    MODEL_PATHS = [
        "gradient_boosting_model.pkl",
        "model.pkl",
        "sglt2_model.pkl",
        "random_forest_model.pkl"
    ]
    
    model = None
    used_path = None
    
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                st.info(f"Attempting to load model from: {model_path}")
                
                # Try joblib first
                try:
                    model = joblib.load(model_path)
                    used_path = model_path
                    st.success(f"✅ Model loaded successfully using joblib: {model_path}")
                    break
                except Exception as e:
                    st.warning(f"Joblib failed for {model_path}: {e}")
                
                # Try pickle as fallback
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    used_path = model_path
                    st.success(f"✅ Model loaded successfully using pickle: {model_path}")
                    break
                except Exception as e:
                    st.warning(f"Pickle failed for {model_path}: {e}")
                    
            except Exception as e:
                st.warning(f"Failed to load {model_path}: {e}")
                continue
    
    if model is None:
        st.error("""
        ❌ Could not load any model file. This is usually due to:
        
        1. **Missing model file** - Ensure you have a trained model file in the app directory
        2. **Version mismatch** - The model was trained with different library versions
        
        **Solutions:**
        - Place your model file (gradient_boosting_model.pkl) in the app directory
        - Or train a new model using the code below
        """)
        
        # Option to use a simple fallback model
        if st.button("🚨 Use Simple Fallback Model (Limited Accuracy)"):
            model = create_fallback_model()
            if model is not None:
                st.success("✅ Simple fallback model created. Predictions will have limited accuracy.")
    
    return model, used_path

def create_fallback_model():
    """Create a simple fallback model when main model fails"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create a dummy model (this would be replaced with actual trained model)
        # For demo purposes only - in practice, you should use a properly trained model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        st.warning("⚠️ Using demo fallback model - predictions are for demonstration only")
        return model
    except Exception as e:
        st.error(f"Failed to create fallback model: {e}")
        return None

@st.cache_resource
def load_features():
    """Load model features with caching"""
    FEATURE_PATHS = [
        "model_features.json",
        "features.json", 
        "selected_features.json"
    ]
    
    for feature_path in FEATURE_PATHS:
        if os.path.exists(feature_path):
            try:
                with open(feature_path, "r") as f:
                    features = json.load(f)
                st.success(f"✅ Features loaded from: {feature_path}")
                return features
            except Exception as e:
                st.warning(f"Failed to load features from {feature_path}: {e}")
                continue
    
    # If no feature file found, create default features
    st.warning("⚠️ No feature file found. Using default molecular descriptors.")
    default_features = [
        'nHBAcc_Lipinski', 'nHBDon_Lipinski', 'MolWt', 'MolLogP', 
        'NumRotatableBonds', 'NumHeteroatoms', 'FractionCsp3',
        'nC', 'nO', 'nN', 'nF', 'nCl', 'nS'
    ]
    return default_features

# Load model and features
model, model_path = load_model_with_fallback()
model_features = load_features()

# Show model info if loaded
if model is not None:
    st.sidebar.success(f"✅ Model: {type(model).__name__}")
    if model_path:
        st.sidebar.info(f"📁 Source: {model_path}")

# -------------------- HELPER FUNCTIONS --------------------
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit"""
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def get_rdkit_descriptors(mol):
    """Calculate basic RDKit descriptors as fallback"""
    desc_dict = {}
    try:
        # Basic molecular descriptors
        desc_dict["MolWt"] = Descriptors.MolWt(mol)
        desc_dict["MolLogP"] = Descriptors.MolLogP(mol)
        desc_dict["NumHDonors"] = Lipinski.NumHDonors(mol)
        desc_dict["NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
        desc_dict["NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
        desc_dict["NumHeteroatoms"] = Lipinski.NumHeteroatoms(mol)
        desc_dict["FractionCsp3"] = Lipinski.FractionCsp3(mol)
        
        # Atom counts
        desc_dict["nC"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        desc_dict["nO"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])
        desc_dict["nN"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        desc_dict["nF"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        desc_dict["nCl"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl'])
        desc_dict["nBr"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
        desc_dict["nS"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'S'])
        desc_dict["nP"] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'P'])
        
    except Exception as e:
        st.warning(f"Some RDKit descriptors failed: {e}")
    
    return desc_dict

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame:
    """Calculate molecular descriptors using Mordred with RDKit fallback"""
    if not RDKIT_AVAILABLE:
        st.error("RDKit is not available. Cannot calculate descriptors.")
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    all_descriptors = {}
    
    # Try Mordred first
    if MORDRED_AVAILABLE and Calculator is not None:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_results = calc(mol)
            
            for desc_name, desc_value in mordred_results.items():
                try:
                    desc_name_str = str(desc_name)
                    if desc_value is None or str(desc_value).lower() in ['nan', 'none', '']:
                        all_descriptors[desc_name_str] = 0.0
                    else:
                        all_descriptors[desc_name_str] = float(desc_value)
                except (ValueError, TypeError):
                    all_descriptors[desc_name_str] = 0.0
                    
        except Exception as e:
            st.warning(f"Mordred calculation had issues: {e}")
    
    # Add RDKit descriptors as fallback
    rdkit_descriptors = get_rdkit_descriptors(mol)
    all_descriptors.update(rdkit_descriptors)
    
    # Fill in missing features
    feature_values = {}
    for feat in features:
        if feat in all_descriptors:
            feature_values[feat] = all_descriptors[feat]
        else:
            # Try to calculate atom counts for patterns like 'nX'
            if feat.startswith('n') and len(feat) > 1:
                element = feat[1:]
                try:
                    count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol().lower() == element.lower()])
                    feature_values[feat] = float(count)
                except Exception:
                    feature_values[feat] = 0.0
            else:
                feature_values[feat] = 0.0
    
    return pd.DataFrame([feature_values], columns=features)

def draw_molecule(smiles: str, size=(400, 400)):
    """Generate molecule image from SMILES"""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None

def create_manual_waterfall(shap_val, base_value, desc_df, feature_names):
    """Create a manual waterfall plot when SHAP's built-in fails"""
    # Sort features by contribution magnitude and take top 10
    features_sorted = sorted(zip(feature_names, shap_val, desc_df.iloc[0].values), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)[:10]
    
    features = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    actual_values = [f[2] for f in features_sorted]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars with colors based on contribution direction
    colors = ['red' if v < 0 else 'blue' for v in values]
    bars = ax.bar(range(len(features)), values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val, actual_val) in enumerate(zip(bars, values, actual_values)):
        height = bar.get_height()
        label_pos = height + (0.01 if height >= 0 else -0.01)
        ax.text(bar.get_x() + bar.get_width()/2., label_pos,
                f'{val:.3f}\n(actual: {actual_val:.2f})', 
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=8, rotation=0)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('SHAP Value (Impact on Prediction)')
    ax.set_title('Top 10 Feature Contributions to Prediction', fontweight='bold')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add baseline and prediction line
    ax.axhline(y=base_value, color='green', linestyle='--', alpha=0.7, label=f'Base Value: {base_value:.3f}')
    final_value = base_value + sum(values)
    ax.axhline(y=final_value, color='red', linestyle='--', alpha=0.7, label=f'Final: {final_value:.3f}')
    ax.legend()
    
    plt.tight_layout()
    return fig

def get_prediction_probability(model, desc_df):
    """Get prediction probability with robust error handling"""
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(desc_df)[0]
            if len(probs) == 2:  # Binary classification
                return float(probs[1])  # Probability of class 1 (active)
            else:
                # Multi-class: find index of positive class
                try:
                    if hasattr(model, 'classes_'):
                        positive_idx = list(model.classes_).index(1)
                        return float(probs[positive_idx])
                    else:
                        # Assume last class is positive if we can't determine
                        return float(probs[-1])
                except (ValueError, IndexError):
                    return float(probs[0])
        return None
    except Exception:
        return None

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

# Dependency status in sidebar
st.sidebar.markdown("### Dependency Status")
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    st.sidebar.markdown(f"RDKit: {'✅' if RDKIT_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"Mordred: {'✅' if MORDRED_AVAILABLE else '❌'}")
with status_col2:
    st.sidebar.markdown(f"SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"Ketcher: {'✅' if KETCHER_AVAILABLE else '❌'}")

if not RDKIT_AVAILABLE:
    st.sidebar.error("RDKit is required but not available. Basic functionality will be limited.")

# Model status
if model is None:
    st.sidebar.error("❌ Model: Not loaded")
else:
    st.sidebar.success(f"✅ Model: Loaded")

input_mode = st.sidebar.radio("Input Method:", ["SMILES String", "Draw Molecule"])

smiles = ""
if input_mode == "SMILES String":
    smiles = st.sidebar.text_area(
        "SMILES Notation:",
        value="",
        height=100,
        placeholder="Enter SMILES string here...\nExample: CCO for ethanol\nExample: C1=CC=CC=C1 for benzene\nExample: CC(=O)O for acetic acid"
    )
    st.sidebar.markdown("💡 **Tip:** Use common SMILES notation for organic molecules")
else:
    if KETCHER_AVAILABLE:
        smiles = st_ketcher("", key="ketcher")
        st.sidebar.markdown("🎨 **Draw** your molecule in the editor above")
    else:
        st.sidebar.warning("Ketcher not available. Using SMILES input.")
        smiles = st.sidebar.text_area(
            "SMILES Notation:",
            value="",
            height=100,
            placeholder="Enter SMILES string here..."
        )

smiles = smiles.strip()

# Example molecules
st.sidebar.markdown("---")
st.sidebar.markdown("### Example Molecules")
example_col1, example_col2 = st.sidebar.columns(2)
with example_col1:
    if st.sidebar.button("Benzene"):
        st.session_state.example_smiles = "C1=CC=CC=C1"
    if st.sidebar.button("Aspirin"):
        st.session_state.example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
with example_col2:
    if st.sidebar.button("Ethanol"):
        st.session_state.example_smiles = "CCO"
    if st.sidebar.button("Caffeine"):
        st.session_state.example_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

# Use example if selected
if hasattr(st.session_state, 'example_smiles'):
    smiles = st.session_state.example_smiles

# -------------------- MAIN PREDICTION AREA --------------------
if not smiles:
    st.info("👈 Enter a SMILES string or draw a molecule in the sidebar to start")
    st.info("💡 Try the example molecules in the sidebar!")
    
    # Show troubleshooting if model failed to load
    if model is None:
        st.markdown("---")
        st.subheader("🚨 Model Loading Issue Detected")
        st.markdown("""
        **To fix the model loading issue:**
        
        1. **Ensure you have a model file** in the app directory (e.g., `gradient_boosting_model.pkl`)
        2. **Check library versions** - the model might have been trained with different versions
        3. **Try retraining** the model with current library versions
        
        **Required files in your app directory:**
        - `gradient_boosting_model.pkl` (or similar)
        - `model_features.json` (list of features used by the model)
        """)
    
    st.stop()

# Validate SMILES
if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string — please check your input")
    st.stop()

# Check if model is available before proceeding
if model is None:
    st.error("""
    ❌ No model available for predictions.
    
    Please ensure you have a trained model file in the app directory or 
    click the fallback model button in the sidebar.
    """)
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
    
    # Basic molecule info
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.write(f"**Molecular Formula:** {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
                st.write(f"**Molecular Weight:** {Descriptors.MolWt(mol):.2f}")
        except Exception:
            pass

with col2:
    st.subheader("🧮 Descriptor Calculation")
    with st.spinner("🔄 Calculating molecular descriptors..."):
        desc_df = calculate_selected_descriptors(smiles, model_features)

    if desc_df is None:
        st.error("❌ Descriptor calculation failed. Check RDKit installation.")
        st.stop()

    # Show descriptors with status
    st.write(f"**Calculated Descriptors ({len(model_features)} total):**")
    desc_display = desc_df.T.rename(columns={0: 'Value'})
    desc_display['Status'] = ['⚠️' if x == 0.0 else '✅' for x in desc_display['Value']]
    
    # Display in expandable section to save space
    with st.expander("View all calculated descriptors", expanded=False):
        st.dataframe(desc_display, use_container_width=True)

    # Show warning if many descriptors are zero
    zero_count = (desc_df.iloc[0] == 0.0).sum()
    if zero_count > len(model_features) * 0.3:
        st.warning(f"⚠️ {zero_count} descriptors ({zero_count/len(model_features)*100:.1f}%) calculated as zero. This may affect prediction accuracy.")

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
            prob = get_prediction_probability(model, desc_df)
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.stop()

    # Display prediction result
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.markdown('<div class="prediction-box active">', unsafe_allow_html=True)
            st.markdown("## 🟢 ACTIVE")
            st.markdown("**Predicted as SGLT2 Inhibitor**")
            if prob is not None:
                st.markdown(f"**Confidence:** {prob:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box inactive">', unsafe_allow_html=True)
            st.markdown("## 🔴 INACTIVE")
            st.markdown("**Not predicted as SGLT2 Inhibitor**")
            if prob is not None:
                st.markdown(f"**Confidence:** {1-prob if prob is not None else 'N/A':.1%}" if prob is not None else "**Confidence:** N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if prob is not None:
            confidence_level = "High confidence" if prob > 0.7 or prob < 0.3 else "Medium confidence" if prob > 0.6 or prob < 0.4 else "Low confidence"
            delta_color = "normal"
            if pred == 1:
                confidence_value = prob
                delta_direction = "inverse" if prob < 0.5 else "normal"
            else:
                confidence_value = 1 - prob
                delta_direction = "inverse" if (1 - prob) < 0.5 else "normal"
            
            st.metric(
                label="**Prediction Confidence**",
                value=f"{confidence_value:.1%}",
                delta=confidence_level,
                delta_color=delta_color
            )
        else:
            st.info("Probability estimation not available for this model")

    # -------------------- SHAP INTERPRETATION --------------------
    if SHAP_AVAILABLE and shap is not None and model is not None:
        st.markdown("---")
        st.subheader("📈 Model Interpretation")
        
        with st.spinner("🔍 Generating SHAP explanation..."):
            try:
                # Create explainer
                explainer = shap.TreeExplainer(model)
                
                # Get SHAP values
                shap_values = explainer.shap_values(desc_df)
                expected_value = explainer.expected_value
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # List format (common for tree models)
                    if len(shap_values) == 2:  # Binary classification
                        shap_val = shap_values[1][0]  # Use class 1 (active)
                        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                    else:
                        shap_val = shap_values[0][0]  # Use first class
                        base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                else:
                    # Array format
                    if len(shap_values.shape) == 3:  # (samples, features, classes)
                        shap_val = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1 else expected_value
                    else:  # (samples, features)
                        shap_val = shap_values[0]
                        base_value = expected_value

                shap_val = np.array(shap_val).flatten()
                
                # Create waterfall plot
                st.subheader("Feature Contribution Analysis")
                
                try:
                    # Use SHAP's built-in waterfall plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.waterfall(shap.Explanation(values=shap_val, 
                                                         base_values=base_value,
                                                         data=desc_df.iloc[0].values,
                                                         feature_names=desc_df.columns.tolist()),
                                       max_display=10, show=False)
                    plt.title("Top 10 Feature Contributions to Prediction", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"Standard SHAP waterfall plot failed: {e}")
                    st.info("Creating alternative visualization...")
                    # Fallback to manual waterfall
                    fig = create_manual_waterfall(shap_val, base_value, desc_df, desc_df.columns.tolist())
                    st.pyplot(fig)
                    plt.close()

                # Feature importance summary
                st.subheader("Feature Importance Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Positive Contributors** (increase activity prediction):")
                    positive_features = sorted(zip(desc_df.columns, shap_val, desc_df.iloc[0].values), 
                                             key=lambda x: x[1], reverse=True)[:5]
                    for feat, shap_val, actual_val in positive_features:
                        if shap_val > 0:
                            st.markdown(f"<span class='feature-positive'>↑ {feat}: +{shap_val:.3f}</span> (value: {actual_val:.2f})", 
                                      unsafe_allow_html=True)
                
                with col2:
                    st.write("**Top Negative Contributors** (decrease activity prediction):")
                    negative_features = sorted(zip(desc_df.columns, shap_val, desc_df.iloc[0].values), 
                                             key=lambda x: x[1])[:5]
                    for feat, shap_val, actual_val in negative_features:
                        if shap_val < 0:
                            st.markdown(f"<span class='feature-negative'>↓ {feat}: {shap_val:.3f}</span> (value: {actual_val:.2f})", 
                                      unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
                st.info("SHAP explanation is not available for this prediction.")
    else:
        st.info("🔍 SHAP interpretation is not available. Install SHAP for model explanation features.")

# -------------------- TROUBLESHOOTING SECTION --------------------
st.markdown("---")
with st.expander("🔧 Troubleshooting & Model Information"):
    st.subheader("Model Information")
    
    if model is not None:
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of Features:** {len(model_features)}")
        
        if hasattr(model, 'n_estimators'):
            st.write(f"**Number of Estimators:** {model.n_estimators}")
    
    st.subheader("Common Issues & Solutions")
    
    st.markdown("""
    **1. Model Loading Errors:**
    - **Cause:** Version mismatch between training and inference environments
    - **Fix:** Retrain model with current library versions or match versions
    
    **2. Missing Dependencies:**
    - RDKit: Required for molecular operations
    - SHAP: Optional for model interpretation
    - Mordred: Optional for advanced descriptors
    
    **3. Descriptor Calculation Issues:**
    - Some descriptors may calculate as zero
    - This is normal for certain molecule types
    - The app uses RDKit fallbacks when Mordred fails
    """)
    
    st.subheader("Required Files")
    st.markdown("""
    - `gradient_boosting_model.pkl` - Trained machine learning model
    - `model_features.json` - List of features used by the model
    """)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧪 <strong>SGLT2 Inhibitor Prediction Tool</strong> | Built with Streamlit, RDKit, and Machine Learning</p>
    <p><small>For research use only. Always validate predictions with experimental data.</small></p>
</div>
""", unsafe_allow_html=True)
