import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample weather dataset
weather_data = {
    'Temperature': [20, 25, 30, 35, 22, 28, 33, 38],
    'Humidity': [40, 50, 60, 70, 45, 55, 65, 75],
    'WindSpeed': [10, 15, 20, 25, 12, 18, 23, 28],
    'Precipitation': [0, 0, 0, 0, 5, 10, 15, 20],
    'Rainfall': [0, 0, 0, 0, 1, 1, 1, 1],
}

weather_df = pd.DataFrame(weather_data)

# Features and target
X = weather_df[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']]
y = weather_df['Rainfall']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI for Weather Prediction
st.title('Weather Prediction')

# User input for predictions
temperature = st.slider('Temperature (°C)', min_value=0, max_value=40, value=20)
humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50)
wind_speed = st.slider('Wind Speed (km/h)', min_value=0, max_value=50, value=15)
precipitation = st.slider('Precipitation (mm)', min_value=0, max_value=50, value=10)

# Weather Prediction
if st.button('Predict Rainfall'):
    rainfall_prediction = model.predict([[temperature, humidity, wind_speed, precipitation]])[0]
    st.write(f"Predicted Rainfall: {rainfall_prediction:.2f} mm")

# Show the sample weather dataset
st.subheader('Sample Weather Dataset')
st.write(weather_df)