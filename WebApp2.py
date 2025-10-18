import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --------------------------------------------
# 1️⃣ Load and preprocess your dataset
# --------------------------------------------
data1 = pd.read_csv('Financial_inclusion_dataset.csv')  # Change to your actual file name

# Encode categorical variables (same as your previous mappings)
data1['country'] = data1['country'].map({'Rwanda':0,'Tanzania':1,'Kenya':2, 'Uganda':3})
data1['cellphone_access'] = data1['cellphone_access'].map({'Yes':0,'No':1})
data1['gender_of_respondent'] = data1['gender_of_respondent'].map({'Female':0,'Male':1})
data1['bank_account'] = data1['bank_account'].map({'Yes':0,'No':1})
data1['marital_status'] = data1['marital_status'].map({
    'Married/Living together':0,'Single/Never Married':1,
    'Widowed':2, 'Divorced/Seperated':3, 'Dont know':4
})
data1['education_level'] = data1['education_level'].map({
    'Primary education':0,'No formal education':1,
    'Secondary education':2, 'Tertiary education':3,
    'Vocational/Specialised training':4, 'Other/Dont know/RTA':5
})
data1['job_type'] = data1['job_type'].map({
    'Self employed':0,'Government Dependent':1,
    'Informally employed':2, 'Farming and Fishing':3,
    'Remittance Dependent':4, 'No Income':5,
    'Formally employed Government':6,
    'Dont Know/Refuse to answer':7,
    'Other Income':8, 'Formally employed Private':9
})
data1['location_type'] = data1['location_type'].map({'Rural':0,'Urban':1})
data1['relationship_with_head'] = data1['relationship_with_head'].map({
    'Head of Household':0,'Spouse':1, 'Child':2,
    'Parent':3, 'Other relative':4, 'Other non-relatives':5
})

# --------------------------------------------
# 2️⃣ Define features and target
# --------------------------------------------
x = data1[['country','cellphone_access','location_type','relationship_with_head',
            'gender_of_respondent','marital_status','education_level','job_type']]
y = data1['bank_account']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train logistic regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(x_train, y_train)

# --------------------------------------------
# 3️⃣ Streamlit Interface
# --------------------------------------------
st.title("Bank Account Prediction App 💳")
st.write("Enter the details below to predict whether a person has a bank account.")

country = st.selectbox('Country', ['Rwanda', 'Tanzania', 'Kenya', 'Uganda'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
location_type = st.selectbox('Location Type', ['Rural', 'Urban'])
relationship_with_head = st.selectbox('Relationship with Head',
    ['Head of Household','Spouse','Child','Parent','Other relative','Other non-relatives'])
gender = st.selectbox('Gender of Respondent', ['Female', 'Male'])
marital_status = st.selectbox('Marital Status',
    ['Married/Living together','Single/Never Married','Widowed','Divorced/Seperated','Dont know'])
education_level = st.selectbox('Education Level',
    ['Primary education','No formal education','Secondary education','Tertiary education',
     'Vocational/Specialised training','Other/Dont know/RTA'])
job_type = st.selectbox('Job Type',
    ['Self employed','Government Dependent','Informally employed','Farming and Fishing',
     'Remittance Dependent','No Income','Formally employed Government',
     'Dont Know/Refuse to answer','Other Income','Formally employed Private'])

# Same mappings
mapping_dicts = {
    'country': {'Rwanda':0,'Tanzania':1,'Kenya':2,'Uganda':3},
    'cellphone_access': {'Yes':0,'No':1},
    'gender_of_respondent': {'Female':0,'Male':1},
    'marital_status': {'Married/Living together':0,'Single/Never Married':1,'Widowed':2,'Divorced/Seperated':3,'Dont know':4},
    'education_level': {'Primary education':0,'No formal education':1,'Secondary education':2,'Tertiary education':3,
                        'Vocational/Specialised training':4,'Other/Dont know/RTA':5},
    'job_type': {'Self employed':0,'Government Dependent':1,'Informally employed':2,'Farming and Fishing':3,
                 'Remittance Dependent':4,'No Income':5,'Formally employed Government':6,
                 'Dont Know/Refuse to answer':7,'Other Income':8,'Formally employed Private':9},
    'location_type': {'Rural':0,'Urban':1},
    'relationship_with_head': {'Head of Household':0,'Spouse':1,'Child':2,'Parent':3,'Other relative':4,'Other non-relatives':5}
}

# Convert to numeric DataFrame
input_data = pd.DataFrame({
    'country': [mapping_dicts['country'][country]],
    'cellphone_access': [mapping_dicts['cellphone_access'][cellphone_access]],
    'location_type': [mapping_dicts['location_type'][location_type]],
    'relationship_with_head': [mapping_dicts['relationship_with_head'][relationship_with_head]],
    'gender_of_respondent': [mapping_dicts['gender_of_respondent'][gender]],
    'marital_status': [mapping_dicts['marital_status'][marital_status]],
    'education_level': [mapping_dicts['education_level'][education_level]],
    'job_type': [mapping_dicts['job_type'][job_type]]
})

# --------------------------------------------
# 4️⃣ Predict on button click
# --------------------------------------------
if st.button('Validate Prediction'):
    prediction = logreg.predict(input_data)[0]
    probability = logreg.predict_proba(input_data)[0][prediction]

    if prediction == 0:
        st.success(f"✅ Likely to HAVE a bank account ({probability:.2%} confidence)")
    else:
        st.error(f"❌ Likely to NOT have a bank account ({probability:.2%} confidence)")
