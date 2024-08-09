import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the Trained Model

model  = tf.keras.models.load_model('model.h5')

# load the encoder and scaler
with open('OHE_geography.pkl','rb') as file:
    OHE_geography=pickle.load(file)
    
with open('LE_gender.pkl','rb') as file:
    LE_gender = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
    
## StreamLit App

st.title("Customer Churn Prediction")

## take inputs 
geography = st.selectbox('Geography', OHE_geography.categories_[0])
gender = st.selectbox('Gender', LE_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [LE_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# separately treating geography data using OHE
geo_encoder = OHE_geography.transform([[geography]])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns=OHE_geography.get_feature_names_out(['Geography']))

# Combine OHE with input data
input_data = pd.concat([input_data.reset_index(drop=True) , geo_encoder_df],axis=1)

# Scale the input Data
input_data_scaled = scaler.transform(input_data)

# predict Churn
prediction = model.predict(input_data_scaled)

st.write("Probability to Churn : ",prediction[0][0])

if prediction[0][0] > 0.5:
    st.write("The Customer is likely to Churn")
else:
    st.write("The Customer is not likely to churn")



