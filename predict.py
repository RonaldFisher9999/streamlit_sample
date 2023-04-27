import torch
import streamlit as st
from model import MyFactorizationMachine
from utils import transform_image
import yaml
from typing import Tuple

@st.cache_data
def load_model() -> MyFactorizationMachine:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = config['args']
    model = MyFactorizationMachine(args).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    
    return model


def get_prediction(model:MyFactorizationMachine, input_data: torch.tensor) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    model.eval()
    y_hat = model.forward(input_data)
    return y_hat.item()
