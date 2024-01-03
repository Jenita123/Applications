import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset (you can replace this with your own dataset)
data = pd.read_csv('Employee.csv')

# Assume 'YearsExperience' as a feature for simplicity
X = data[['YearsExperience']]
y = data['Salary']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI for Salary Prediction
st.title('Salary Prediction')

# User input for predictions
years_experience = st.slider('Years of Experience', min_value=0, max_value=30, value=5)

# Salary Prediction
if st.button('Predict Salary'):
    salary_prediction = model.predict([[years_experience]])[0]
    st.write(f"Predicted Salary: ${salary_prediction:.2f}")