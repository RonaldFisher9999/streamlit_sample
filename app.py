import streamlit as st
import yaml
import torch
from predict import load_model, get_prediction

st.title("Book Rating Prediction Model")

with open("idx.yaml") as f:
    idx = yaml.load(f, Loader=yaml.FullLoader)


col1, col2 = st.columns(2)

age = col1.selectbox('Select your age!',
                      idx['age_bin'].keys())
age_value = int(idx['age_bin'][age])

country = col2.selectbox('Select your country!',
                      idx['country'].keys())
country_value = int(idx['country'][country])


col3, col4 = st.columns(2)

pub_year = col3.selectbox('Select published year of the book!',
                      idx['year_bin'].keys())
pub_value = int(idx['year_bin'][pub_year])

category = col4.selectbox('Select category of the book!',
                      idx['major_cat'].keys())
cat_value = int(idx['major_cat'][category])


input_data = [age_value, country_value, pub_value, cat_value]

def load_predict(input_data) :
    
    input_data = torch.tensor(input_data)
    model = load_model()
    y_hat = get_prediction(model, input_data)
    
    return y_hat

st.header(f"Predicted Rating : {load_predict(input_data):.2f}")    
