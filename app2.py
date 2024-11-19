import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model and scaler
model = tf.keras.models.load_model('ann_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set background (optional)
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("https://leftshiftone.com/wp-content/uploads/2022/03/churn-prediction-banner02.jpg")

# App title and instructions
st.title("Customer Churn Prediction using ANN🤖")
st.write("🔍 Discover if your customer is likely to stay or leave—predict churn with just a few details! 📊")

# Input features for prediction
st.sidebar.header("Input Customer Details")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
geography = st.sidebar.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
age = st.sidebar.slider("Age", 18, 100, 40)
tenue = st.sidebar.number_input("Tenue",min_value=0.0,value=2.0,step=1.0)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", (1, 0))
is_active_member = st.sidebar.selectbox("Is Active Member", (1, 0))
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Process input features
def preprocess_input(credit_score, geography, gender, age, balance, num_of_products, has_cr_card, is_active_member, estimated_salary,tenue):
    # Encode Geography and Gender to match training data encoding
    geography_one_hot = [1, 0, 0] if geography == "France" else [0, 1, 0] if geography == "Germany" else [0, 0, 1]
    gender = 1 if gender == "Male" else 0
    
    # Concatenate all features into a 12-feature array
    inputs = np.array([credit_score] + geography_one_hot + [gender, age, balance, num_of_products, has_cr_card, is_active_member, estimated_salary,tenue])
    
    # Reshape inputs for scaling (needed as a 2D array)
    inputs = inputs.reshape(1, -1)
    
    # Scale the inputs using the loaded scaler
    scaled_input = scaler.transform(inputs)
    return scaled_input

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Predict churn
if st.button("Predict Churn 🔮"):
    input_data = preprocess_input(credit_score, geography, gender, age, balance, num_of_products, has_cr_card, is_active_member, estimated_salary,tenue)
    prediction = model.predict(input_data)
    churn_prediction = "Custmor is likely to Churn.❌" if prediction > 0.5 else "Custmor is NOT likely to Churn✅."
    
    # Save the result in session state to avoid repeated predictions
    st.session_state.prediction = (churn_prediction, prediction[0][0])

# Display the prediction only if it's set in session state
if st.session_state.prediction:
    churn_prediction, prediction_prob = st.session_state.prediction
    st.write(f"**{churn_prediction}**")
    #st.write(f"Prediction Probability: **{prediction_prob:.2f}**")
