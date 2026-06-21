from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import torch
import json
import os
from utils import *




st.title("Sistema de Diagnóstico Inteligente")
st.write("Sube una imagen de la hoja para obtener la predicción del modelo.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png","webp"]
)



@st.cache_resource
def load_model():
    
    model = CNN()
    checkpoint = torch.load('mobilenet_checkpointv2.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'],map_location = "cpu")
    
    
   
   
    return model

model = load_model()

# ---------------------------
# PREDICCIÓN
# ---------------------------


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
   
    

    fig,idx = predict(
    img=image,
    model=model,
    transforms=val_transform,
    device='cpu'
)


    st.subheader("Resultado")
    st.pyplot(fig)


    recomendaciones = obtener_recomendaciones(idx)
    st.markdown("---")
    st.markdown(recomendaciones)



    
   
