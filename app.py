
import streamlit as st
import pandas as pd
import os
import sys
from InsuranceClaimPredictionProject.utils.main_utils import load_obj, transform_dates_and_csl
from InsuranceClaimPredictionProject.constants import Target_Column
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException

# Page configuration
st.set_page_config(
    page_title="Insurance Claim Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom Styling (CSS) - Dark Obsidian & Electric Sapphire Glass Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide default Streamlit sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Main container background - Pitch Black & Radial Indigo/Cyan Glow */
    .stApp {
        background: #060812;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.22) 0%, rgba(6, 8, 18, 1) 70%),
            radial-gradient(circle at 100% 100%, rgba(14, 165, 233, 0.15) 0%, rgba(6, 8, 18, 1) 50%);
        color: #f4f4f5;
    }
    
    /* Hero Banner Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(10, 15, 30, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 50px -15px rgba(99, 102, 241, 0.3);
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 9999px;
        background: linear-gradient(90deg, #4f46e5, #0284c7);
        color: #ffffff;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 15px 0 rgba(79, 70, 229, 0.5);
        border: 1px solid rgba(129, 140, 248, 0.4);
    }
    
    .hero-header-flex {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #93c5fd 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.6rem;
        margin-bottom: 0rem;
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.65);
        box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.35);
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result Cards */
    .result-fraud {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.22) 0%, rgba(127, 29, 29, 0.35) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.8);
        border-radius: 18px;
        padding: 1.8rem;
        margin-top: 1rem;
        box-shadow: 0 15px 35px -10px rgba(220, 38, 38, 0.5);
    }
    
    .result-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.7);
        border-radius: 18px;
        padding: 1.8rem;
        margin-top: 1rem;
        box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.4);
    }
    
    .result-title-fraud {
        color: #fca5a5;
        font-size: 1.6rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-title-genuine {
        color: #6ee7b7;
        font-size: 1.6rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Customizing Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.85);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4f46e5, #0284c7) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.5) !important;
    }

    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #0284c7 100%);
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.65rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(129, 140, 248, 0.4);
        box-shadow: 0 8px 25px -5px rgba(79, 70, 229, 0.5);
        transition: all 0.2s ease;
        width: 100%;
        white-space: nowrap;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -5px rgba(56, 189, 248, 0.6);
        background: linear-gradient(90deg, #6366f1 0%, #0369a1 100%);
    }

    /* Slider track & input field accents */
    div[data-baseweb="slider"] {
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Backend Model & Raw CSV
model_path = os.path.join('final_model', 'model.pkl')
model = load_obj(model_path)

csv_path = os.path.join('data', 'raw', 'insurance_claims.csv')
df_raw = pd.read_csv(csv_path)

# Prepare dataset feature columns
df = df_raw.copy()
drop_cols = ['policy_number', '_c39', Target_Column]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
FEATURE_ORDER = df.columns.tolist()

# Hero Banner Header with High-Tech Vector Shield Logo
st.markdown("""
<div class="hero-container">
    <div class="badge">AI-POWERED FRAUD DETECTION ENGINE</div>
    <div class="hero-header-flex">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.8));">
            <path d="M12 2L4 5V11.09C4 16.14 7.41 20.85 12 22C16.59 20.85 20 16.14 20 11.09V5L12 2Z" fill="url(#shield_grad_blue)" stroke="#38bdf8" stroke-width="1.5"/>
            <path d="M12 6.5L15.5 11.5H13V16.5H11V11.5H8.5L12 6.5Z" fill="#ffffff" opacity="0.95"/>
            <defs>
                <linearGradient id="shield_grad_blue" x1="4" y1="2" x2="20" y2="22" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#6366f1"/>
                    <stop offset="1" stop-color="#0284c7"/>
                </linearGradient>
            </defs>
        </svg>
        <h1 class="hero-title">Insurance Claim Truthfulness Analytics</h1>
    </div>
    <p class="hero-subtitle">Comprehensive ML-driven risk evaluation platform for auditing vehicle and property insurance claims.</p>
</div>
""", unsafe_allow_html=True)

# Stat Cards Summary Row
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""<div class="stat-card"><div class="stat-val">1,000+</div><div class="stat-label">Historical Claims</div></div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="stat-card"><div class="stat-val">CatBoost</div><div class="stat-label">ML Architecture</div></div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="stat-card"><div class="stat-val">38</div><div class="stat-label">Risk Features</div></div>""", unsafe_allow_html=True)
with s4:
    st.markdown("""<div class="stat-card"><div class="stat-val">&lt; 50 ms</div><div class="stat-label">Inference Time</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick Presets Header Bar
st.markdown("##### ⚡ Quick Load Presets")
preset_c1, preset_c2, preset_c3, preset_c4 = st.columns([2, 1, 1, 1])
with preset_c1:
    st.caption("Instantly populate form fields with real historical claim scenarios for testing:")
with preset_c2:
    if st.button("🚨 Fraud Sample", key="btn_fraud_top"):
        fraud_idx = df_raw[df_raw[Target_Column] == 'Y'].index[0] if Target_Column in df_raw.columns else 0
        for col in FEATURE_ORDER:
            st.session_state[f"input_{col}"] = df.iloc[fraud_idx][col]
        st.toast("Loaded High Risk (Fraud) Claim Sample!", icon="🚨")

with preset_c3:
    if st.button("✅ Genuine Sample", key="btn_genuine_top"):
        genuine_idx = df_raw[df_raw[Target_Column] == 'N'].index[0] if Target_Column in df_raw.columns else 0
        for col in FEATURE_ORDER:
            st.session_state[f"input_{col}"] = df.iloc[genuine_idx][col]
        st.toast("Loaded Low Risk (Genuine) Claim Sample!", icon="✅")

with preset_c4:
    if st.button("🔄 Reset Defaults", key="btn_reset_top"):
        for col in FEATURE_ORDER:
            if f"input_{col}" in st.session_state:
                del st.session_state[f"input_{col}"]
        st.toast("Reset fields to default mean values!", icon="🔄")

st.markdown("<br>", unsafe_allow_html=True)

# Categorized Feature Sets
policy_features = ['months_as_customer', 'age', 'policy_bind_date', 'policy_state', 'policy_csl', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip']
demographic_features = ['insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship']
incident_features = ['incident_date', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city', 'incident_location', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses', 'police_report_available']
claim_vehicle_features = ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'capital-gains', 'capital-loss', 'auto_make', 'auto_model', 'auto_year']

st.subheader("📋 Enter Claim & Insured Information")

user_data = {}

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Insured & Policy Profile",
    "💥 Incident & Location",
    "💰 Financial Claims & Vehicle",
    "🔍 Summary Overview"
])

def render_field(col_name):
    session_key = f"input_{col_name}"
    if df[col_name].dtype == 'object':
        options = sorted(df[col_name].dropna().unique())
        default_idx = 0
        if session_key in st.session_state:
            val = st.session_state[session_key]
            if val in options:
                default_idx = options.index(val)
        return st.selectbox(f"{col_name.replace('_', ' ').title()}:", options, index=default_idx, key=session_key)
    else:
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        mean_val = float(df[col_name].mean())
        default_val = mean_val
        if session_key in st.session_state:
            default_val = float(st.session_state[session_key])
            default_val = max(min_val, min(max_val, default_val))
            
        return st.slider(f"{col_name.replace('_', ' ').title()}:", min_value=min_val, max_value=max_val, value=default_val, step=1.0, key=session_key)

with tab1:
    st.markdown("##### 📜 Policy Holder & Demographics")
    c1, c2 = st.columns(2)
    features_tab1 = policy_features + demographic_features
    for idx, col_name in enumerate(features_tab1):
        if col_name in FEATURE_ORDER:
            target_col = c1 if idx % 2 == 0 else c2
            with target_col:
                user_data[col_name] = render_field(col_name)

with tab2:
    st.markdown("##### 📍 Incident Occurrence & Location Details")
    c1, c2 = st.columns(2)
    for idx, col_name in enumerate(incident_features):
        if col_name in FEATURE_ORDER:
            target_col = c1 if idx % 2 == 0 else c2
            with target_col:
                user_data[col_name] = render_field(col_name)

with tab3:
    st.markdown("##### 💵 Financial Claim Amounts & Vehicle Details")
    c1, c2 = st.columns(2)
    for idx, col_name in enumerate(claim_vehicle_features):
        if col_name in FEATURE_ORDER:
            target_col = c1 if idx % 2 == 0 else c2
            with target_col:
                user_data[col_name] = render_field(col_name)

# Fill any remaining features in FEATURE_ORDER if not explicitly listed in tabs
for col_name in FEATURE_ORDER:
    if col_name not in user_data:
        user_data[col_name] = render_field(col_name)

with tab4:
    st.markdown("##### 📋 Input Parameter Matrix")
    st.write("Review input parameters before running inference:")
    preview_df = pd.DataFrame([
        {"Feature": col.replace('_', ' ').title(), "Value": str(user_data.get(col, ''))}
        for col in FEATURE_ORDER
    ])
    st.dataframe(preview_df, use_container_width=True, height=350)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Trigger Button
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_clicked = st.button("🚀 Analyze Claim & Predict Fraud Risk")

if predict_clicked:
    try:
        with st.spinner("Processing input vectors & evaluating model decision boundary..."):
            ordered_user_data = {col: user_data[col] for col in FEATURE_ORDER}
            input_df = pd.DataFrame([ordered_user_data])
            input_df = transform_dates_and_csl(input_df)
            prediction = model.predict(input_df)

        st.markdown("---")
        st.subheader("📢 Assessment Result")

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-fraud">
                <div class="result-title-fraud">
                    <span>🚨 PREDICTION: FRAUDULENT CLAIM DETECTED</span>
                </div>
                <p style="color: #fca5a5; font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 1.2rem;">
                    <strong>High Risk Level:</strong> The predictive model identified significant anomaly indicators matching historical fraudulent claim patterns.
                </p>
                <div style="background: rgba(0, 0, 0, 0.4); padding: 1rem; border-radius: 10px; border-left: 4px solid #ef4444;">
                    <span style="color: #ffffff; font-weight: 700;">Recommended Next Actions:</span>
                    <ul style="color: #e4e4e7; margin-top: 0.5rem; margin-bottom: 0;">
                        <li>Flag claim for Special Investigation Unit (SIU) deep audit.</li>
                        <li>Request physical vehicle inspection and witness statements verification.</li>
                        <li>Cross-check police report availability and injury medical bills.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-genuine">
                <div class="result-title-genuine">
                    <span>✅ PREDICTION: GENUINE CLAIM (NOT FRAUD)</span>
                </div>
                <p style="color: #6ee7b7; font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 1.2rem;">
                    <strong>Low Risk Level:</strong> The claim attributes align cleanly with legitimate policy claim profiles. No major risk anomalies detected.
                </p>
                <div style="background: rgba(0, 0, 0, 0.4); padding: 1rem; border-radius: 10px; border-left: 4px solid #10b981;">
                    <span style="color: #ffffff; font-weight: 700;">Recommended Next Actions:</span>
                    <ul style="color: #e4e4e7; margin-top: 0.5rem; margin-bottom: 0;">
                        <li>Fast-track claim for standard payout processing.</li>
                        <li>Archive claim audit log for routine compliance.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Claim Financial Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Claim Amount", f"${user_data.get('total_claim_amount', 0):,.2f}")
        m2.metric("Vehicle Claim", f"${user_data.get('vehicle_claim', 0):,.2f}")
        m3.metric("Injury Claim", f"${user_data.get('injury_claim', 0):,.2f}")
        m4.metric("Property Claim", f"${user_data.get('property_claim', 0):,.2f}")

    except Exception as e:
        raise ClaimPredictionException(e, sys)
