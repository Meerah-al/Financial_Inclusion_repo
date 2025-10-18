import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('logistic_model.pkl')

st.title("Bank Account Prediction App 💳")
st.write("Enter the details below to predict whether a person has a bank account.")

# Create input fields for all features
country = st.selectbox('Country', ['Rwanda', 'Tanzania', 'Kenya', 'Uganda'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
location_type = st.selectbox('Location Type', ['Rural', 'Urban'])
relationship_with_head = st.selectbox('Relationship with Head',
    ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'])
gender = st.selectbox('Gender of Respondent', ['Female', 'Male'])
marital_status = st.selectbox('Marital Status',
    ['Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated', 'Dont know'])
education_level = st.selectbox('Education Level',
    ['Primary education', 'No formal education', 'Secondary education', 'Tertiary education',
     'Vocational/Specialised training', 'Other/Dont know/RTA'])
job_type = st.selectbox('Job Type',
    ['Self employed', 'Government Dependent', 'Informally employed', 'Farming and Fishing',
     'Remittance Dependent', 'No Income', 'Formally employed Government',
     'Dont Know/Refuse to answer', 'Other Income', 'Formally employed Private'])

# Map categorical values to numeric (same as in your training)
mapping_dicts = {
    'country': {'Rwanda':0,'Tanzania':1,'Kenya':2, 'Uganda':3},
    'cellphone_access': {'Yes':0,'No':1},
    'gender_of_respondent': {'Female':0,'Male':1},
    'marital_status': {'Married/Living together':0,'Single/Never Married':1, 'Widowed':2, 'Divorced/Seperated':3, 'Dont know':4},
    'education_level': {'Primary education':0,'No formal education':1, 'Secondary education':2, 'Tertiary education':3, 'Vocational/Specialised training':4, 'Other/Dont know/RTA':5},
    'job_type': {'Self employed':0,'Government Dependent':1, 'Informally employed':2, 'Farming and Fishing':3, 'Remittance Dependent':4, 'No Income':5, 'Formally employed Government':6, 'Dont Know/Refuse to answer':7, 'Other Income':8, 'Formally employed Private':9},
    'location_type': {'Rural':0,'Urban':1},
    'relationship_with_head': {'Head of Household':0,'Spouse':1, 'Child':2, 'Parent':3, 'Other relative':4, 'Other non-relatives':5}
}

# Convert user input into model format
input_data = pd.DataFrame({
    'country': [mapping_dicts['country'][country]],
    'cellphone_access': [mapping_dicts['cellphone_access'][cellphone_access]],
    'location_type': [mapping_dicts['location_type'][location_type]],
    'relationship_with_head': [mapping_dicts['relationship_with_head'][relationship_with_head]],
    'gender_of_respondent': [mapping_dicts['gender_of_respondent'][gender]],
    'marital_status': [mapping_dicts['marital_status'][marital_status]],
    'education_level': [mapping_dicts['education_level'][education_level]],
    'job_type': [mapping_dicts['job_type'][job_type]],
})

# Button to validate
if st.button('Validate Prediction'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 0:
        st.success(f"✅ This person is likely to **have a bank account.** ({probability:.2%} confidence)")
    else:
        st.error(f"❌ This person is likely to **NOT have a bank account.** ({probability:.2%} confidence)")
