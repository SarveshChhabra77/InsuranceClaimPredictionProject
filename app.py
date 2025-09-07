import streamlit as st
import pandas as pd
import os
import sys
from InsuranceClaimPredictionProject.utils.main_utils import load_obj,transform_dates_and_csl
from InsuranceClaimPredictionProject.constants import Target_Column
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException

# Load model
model_path = os.path.join('final_model', 'model.pkl')
model = load_obj(model_path)


# Load CSV
csv_path = os.path.join('data','raw','insurance_claims.csv')
df = pd.read_csv(csv_path)

# Dropping unwanted columns
drop_cols = ['policy_number' , '_c39',Target_Column]
df.drop(columns=drop_cols,inplace=True)


# Maintain column order based on CSV
FEATURE_ORDER = df.columns.tolist()

st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="✅",
    layout="wide"
)

st.title("📊 Insurance Claim Truthfulness Detector")
st.caption("Provide claim details below to predict whether the claim is **Fraud** or **Not Fraud**")

# Collect user input
user_data = {}

for col in FEATURE_ORDER:
    if df[col].dtype == 'object':  # categorical column
        options = sorted(df[col].dropna().unique())
        user_data[col] = st.selectbox(f"Select {col}:", options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        user_data[col] = st.slider(
            f"Enter {col}:",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1.0
        )

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_data])
        input_df = transform_dates_and_csl(input_df)
        prediction = model.predict(input_df)
        st.markdown("---")
        st.subheader("📢 Prediction Result")
        if prediction[0] == 1:
            st.markdown(
                "<h3 style='color:red;'>❌ Prediction: Fraud</h3>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:green;'>✅ Prediction: Not Fraud</h3>", 
                unsafe_allow_html=True
            )
    except Exception as e:
        raise ClaimPredictionException(e,sys)
