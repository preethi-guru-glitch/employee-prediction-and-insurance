# =========================================================
# HR EMPLOYEE ATTRITION PREDICTION - STREAMLIT APP
# =========================================================

# Run using:
# streamlit run app.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE TITLE
# =========================================================

st.title("HR Employee Attrition Prediction")

st.write("Predict whether employee will leave company or not")

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("HR_comma_sep.csv")

# =========================================================
# SHOW DATASET
# =========================================================

st.subheader("Dataset")

st.dataframe(data.head())

# =========================================================
# ENCODE COLUMNS
# =========================================================

le_salary = LabelEncoder()

data['salary'] = le_salary.fit_transform(data['salary'])

le_department = LabelEncoder()

data['Department'] = le_department.fit_transform(data['Department'])

# =========================================================
# REMOVE WEAK COLUMNS
# =========================================================

X = data.drop(
    ['left', 'last_evaluation', 'Department'],
    axis=1
)

y = data['left']

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# =========================================================
# MODEL ACCURACY
# =========================================================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy : {accuracy:.2f}")

# =========================================================
# VISUALIZATION
# =========================================================

st.subheader("Scatter Plot")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    data['satisfaction_level'],
    data['average_montly_hours'],
    c=data['left'],
    cmap='coolwarm'
)

ax.set_xlabel("Satisfaction Level")
ax.set_ylabel("Average Monthly Hours")

ax.set_title("Satisfaction vs Monthly Hours")

plt.colorbar(scatter)

st.pyplot(fig)

# =========================================================
# USER INPUT
# =========================================================

st.subheader("Employee Details Prediction")

satisfaction_level = st.slider(
    "Satisfaction Level",
    0.0,
    1.0,
    0.5
)

number_project = st.slider(
    "Number of Projects",
    1,
    10,
    3
)

average_montly_hours = st.slider(
    "Average Monthly Hours",
    50,
    350,
    200
)

time_spend_company = st.slider(
    "Years in Company",
    1,
    10,
    3
)

Work_accident = st.selectbox(
    "Work Accident",
    [0,1]
)

promotion_last_5years = st.selectbox(
    "Promotion in Last 5 Years",
    [0,1]
)

salary = st.selectbox(
    "Salary",
    ['low', 'medium', 'high']
)

# =========================================================
# ENCODE SALARY INPUT
# =========================================================

salary_encoded = le_salary.transform([salary])[0]

# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = [[
    satisfaction_level,
    number_project,
    average_montly_hours,
    time_spend_company,
    Work_accident,
    promotion_last_5years,
    salary_encoded
]]

# =========================================================
# SCALE INPUT
# =========================================================

input_scaled = scaler.transform(input_data)

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict"):

    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("Employee may leave company")

    else:
        st.success("Employee may stay in company")