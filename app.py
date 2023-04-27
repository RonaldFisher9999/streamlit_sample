import streamlit as st
import yaml
import torch
from predict import load_model, get_prediction

st.title("Book Rating Prediction Model")

with open("idx.yaml") as f:
    idx = yaml.load(f, Loader=yaml.FullLoader)
# st.write(idx.keys())

age = st.selectbox('Please select age!',
                      idx['age_bin'].keys())
st.write('Your Age:', age)
st.write(idx['age_bin'][age])
age_value = int(idx['age_bin'][age])


country = st.selectbox('Please select country!',
                      idx['country'].keys())
st.write('Your Country:', country)
st.write(idx['country'][country])
country_value = int(idx['country'][country])


pub_year = st.selectbox('Please select published year!',
                      idx['year_bin'].keys())
st.write('Your Published Year:', pub_year)
st.write(idx['year_bin'][pub_year])
pub_value = int(idx['year_bin'][pub_year])


category = st.selectbox('Please select category!',
                      idx['major_cat'].keys())
st.write('Your Category:', category)
st.write(idx['major_cat'][category])
cat_value = int(idx['major_cat'][category])


input_data = [age_value, country_value, pub_value, cat_value]
# st.write(input_data)

def load_predict(input_data) :
    
    input_data = torch.tensor(input_data)
    model = load_model()
    y_hat = get_prediction(model, input_data)
    
    return y_hat

st.header(f"Predicted Rating : {load_predict(input_data)}")    
