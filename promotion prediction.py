import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Creating an example dataset for promotion prediction
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 28, 33],
    'RelationshipStatus': ['Single', 'Single', 'In a relationship', 'Married', 'Single', 'In a relationship'],
    'EducationLevel': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'Occupation': ['Engineer', 'Doctor', 'Teacher', 'Artist', 'Engineer', 'Lawyer'],
    'Lifestyle': ['Active', 'Active', 'Sedentary', 'Sedentary', 'Active', 'Sedentary'],
    'Married': [0, 0, 1, 1, 0, 1],  # 1 for married, 0 for not married
    'Promoted': [0, 1, 1, 0, 0, 1]  # 1 for promoted, 0 for not promoted
})

# Convert non-numeric values to numeric
relationship_mapping = {'Single': 1, 'In a relationship': 2, 'Married': 3}
education_mapping = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
occupation_mapping = {'Engineer': 1, 'Doctor': 2, 'Teacher': 3, 'Artist': 4, 'Lawyer': 5}
lifestyle_mapping = {'Active': 1, 'Sedentary': 2}

data['RelationshipStatus'] = data['RelationshipStatus'].map(relationship_mapping)
data['EducationLevel'] = data['EducationLevel'].map(education_mapping)
data['Occupation'] = data['Occupation'].map(occupation_mapping)
data['Lifestyle'] = data['Lifestyle'].map(lifestyle_mapping)

# Splitting the data
X = data[['Age', 'RelationshipStatus', 'EducationLevel', 'Occupation', 'Lifestyle', 'Married']]
y_promotion = data['Promoted']
X_train, X_test, y_train_promotion, y_test_promotion = train_test_split(X, y_promotion, test_size=0.2, random_state=42)

# Model Training for Promotion
promotion_model = LogisticRegression()
promotion_model.fit(X_train, y_train_promotion)

# Streamlit UI for Promotion Prediction
st.title('Promotion Prediction')

# User input for predictions
age = st.slider('Age', min_value=18, max_value=60, value=25)
relationship_status = st.selectbox('Relationship Status', ['Single', 'In a relationship', 'Married'])
education_level = st.selectbox('Education Level', ['Bachelor', 'Master', 'PhD'])
occupation = st.selectbox('Occupation', ['Engineer', 'Doctor', 'Teacher', 'Artist', 'Lawyer'])
lifestyle = st.selectbox('Lifestyle', ['Active', 'Sedentary'])
married = st.radio('Married', [0, 1])

relationship_status_num = relationship_mapping[relationship_status]
education_num = education_mapping[education_level]
occupation_num = occupation_mapping[occupation]
lifestyle_num = lifestyle_mapping[lifestyle]

# Promotion Prediction
if st.button('Predict Promotion'):
    promotion_prediction = promotion_model.predict([[age, relationship_status_num, education_num, occupation_num, lifestyle_num, married]])[0]
    promotion_status = 'Promoted' if promotion_prediction == 1 else 'Not Promoted'
    st.write(f"Predicted Promotion Status: {promotion_status}")