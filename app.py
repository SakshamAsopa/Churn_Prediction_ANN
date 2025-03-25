import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehotencoder_geo.pkl', 'rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app with improved UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Predict the likelihood of a customer churning based on their profile.</p>", unsafe_allow_html=True)

st.divider()  # Adds a horizontal line for better separation

# Create two columns for better UI organization
col1, col2 = st.columns(2)

# User Input Fields
with col1:
    geography = st.selectbox('🌍 Geography', onehotencoder_geo.categories_[0], help="Select customer's country.")
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_, help="Select customer's gender.")
    age = st.slider("🎂 Age", 18, 99, 35, help="Select the age of the customer.")
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=600, step=10, help="Enter customer's credit score.")
    balance = st.number_input('🏦 Balance', min_value=0.0, max_value=500000.0, value=10000.0, step=100.0, help="Enter account balance.")

with col2:
    estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0, step=500.0, help="Enter estimated salary.")
    tenure = st.slider('📆 Tenure (Years)', 0, 10, 5, help="Select the number of years with the bank.")
    num_of_prod = st.slider('📦 Number of Products', 1, 4, 1, help="Select the number of products owned by the customer.")
    has_cr_card = st.radio("💳 Has Credit Card?", [0, 1], help="Does the customer own a credit card?")
    is_active_mem = st.radio("📢 Active Member?", [0, 1], help="Is the customer an active member of the bank?")

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# OneHotEncode Geography
geo_encoded = onehotencoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

# Combine input features
input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_prod],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_mem],
        'EstimatedSalary': [estimated_salary]
    }
)

# Concatenate with encoded geography data
input_data = pd.concat([input_data, geo_df], axis=1)

# Predict Churn on Button Click
if st.button("🚀 Predict Churn", use_container_width=True):
    with st.spinner("Analyzing data... Please wait."):
        # Scale input data
        inp_data_scaled = scaler.transform(input_data)

        # Make prediction
        pred = model.predict(inp_data_scaled)
        prob = float(pred[0][0])  # Ensure correct float conversion

        # Display Results
        st.divider()
        st.subheader("🔮 Prediction Result")
        st.metric(label="Churn Probability", value=f"{prob:.2%}")

        if prob > 0.5:
            st.error("⚠️ The customer is **likely to churn**. Consider engagement strategies.", icon="⚠️")
        else:
            st.success("✅ The customer is **not likely to churn**. Keep up the good relationship!", icon="✅")

st.markdown("<p style='text-align: center; color: #888;'>© 2025 - Powered by AI</p>", unsafe_allow_html=True)
