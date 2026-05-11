# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load Dataset
data = pd.read_csv("insurance_data.csv")

# Input and Output
X = data[['age']]
y = data['bought_insurance']

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("Insurance Prediction App")

st.write("Predict whether a person will buy insurance or not.")

# User Input
age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)

# Predict Button
if st.button("Predict"):

    # Prediction
    prediction = model.predict([[age]])

    # Probability using Sigmoid
    probability = model.predict_proba([[age]])[0][1]

    st.write("Sigmoid Probability :", probability)

    if prediction[0] == 1:
        st.success("Person will Buy Insurance")
    else:
        st.error("Person will NOT Buy Insurance")