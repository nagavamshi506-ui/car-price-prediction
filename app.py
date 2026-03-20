import streamlit as st
import pandas as pd
import pickle
import os

# load model and columns
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(BASE_DIR,'model.pkl')
columns_path=os.path.join(BASE_DIR,"columns.pkl")
with open(model_path,"rb") as f:
    model = pickle.load(f)
with open(columns_path,"rb") as f:
    X_columns = pickle.load(f)
    
st.title("🚗 Car Price Prediction App")

# INPUTS (UI instead of input())
brand = st.text_input("Enter car brand")
model_name = st.text_input("Enter car model")
year = st.number_input("Enter year", 1990, 2025)
fuel = st.selectbox("Fuel type", ["petrol", "diesel", "cng", "electric"])
transmission = st.selectbox("Transmission", ["manual", "automatic"])
engine = st.number_input("Engine CC")
mileage = st.number_input("Mileage (kmpl)")
seats = st.number_input("Seating capacity", 1, 10)
safety = st.slider("Safety rating", 0.0, 5.0)
body = st.selectbox("Body type", ["hatchback", "sedan", "suv", "muv"])

# PREDICT BUTTON
if st.button("Predict Price"):

    user_data = {
        "brand": brand,
        "model": model_name,
        "year": year,
        "fuel_type": fuel,
        "transmission": transmission,
        "engine_cc": engine,
        "mileage_kmpl": mileage,
        "seating_capacity": seats,
        "safety_rating": safety,
        "body_type": body
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # One-hot encoding
    user_df = pd.get_dummies(user_df)

    # Match training columns
    user_df = user_df.reindex(columns=X_columns, fill_value=0)

    # Prediction
    prediction = model.predict(user_df)

    st.success(f"💰 Predicted Price: ₹ {prediction[0]:,.2f}")