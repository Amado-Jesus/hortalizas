import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
import os


from torchvision import transforms



val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

def CNN():

    model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    

    for param in model.parameters():
        
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
        

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    in_features = model.classifier[0].in_features
    

    model.classifier = nn.Sequential(
        
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Hardswish(),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.Hardswish(),
        nn.Dropout(0.3),

        nn.Linear(256, 19)

    )

    return model
    
with open("recomendaciones.json", "r", encoding="utf-8") as archivo:
    
    enfermedades = json.load(archivo)   


idx = {
    0: 'Maíz___Mancha_foliar_por_Cercospora_(Mancha_gris)',
    1: 'Maíz___Roya_común',
    2: 'Maíz___Tizón_foliar_del_norte',
    3: 'Maíz___Sano',

    4: 'Pimiento_morrón___Mancha_bacteriana',
    5: 'Pimiento_morrón___Sano',

    6: 'Papa___Tizón_temprano',
    7: 'Papa___Tizón_tardío',
    8: 'Papa___Sana',

    9: 'Tomate___Mancha_bacteriana',
    10: 'Tomate___Tizón_temprano',
    11: 'Tomate___Tizón_tardío',
    12: 'Tomate___Moho_de_la_hoja',
    13: 'Tomate___Mancha_foliar_por_Septoria',
    14: 'Tomate___Ácaro_de_dos_puntos',
    15: 'Tomate___Mancha_diana',
    16: 'Tomate___Virus_del_rizado_amarillo_de_la_hoja_del_tomate',
    17: 'Tomate___Virus_del_mosaico_del_tomate',
    18: 'Tomate___Sano'
}

def predict(img, model,transforms, device='cpu'):
    """
    Predice la clase de una imagen y muestra visualización con probabilidades
    
    Args:
        x: ruta a la imagen
        model: modelo entrenado
        transforms: transformaciones de preprocesamiento
        idx: diccionario {indice: nombre_clase}
        device: 'cpu' o 'cuda'
    """
    
    # Cargar y preprocesar imagen
   # img = Image.open(x).convert("RGB")
    x_tensor = val_transform(img).unsqueeze(0).to(device)
    
    # Predicción
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    # Obtener predicción
    pred_idx = int(np.argmax(probs))
    pred_clase = idx[pred_idx]
    confidence = probs[pred_idx] * 100
    
    # Ordenar probabilidades de mayor a menor
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_classes = [idx[i] for i in sorted_indices]
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.patch.set_facecolor('white')
    
    # 📷 IZQUIERDA: Imagen original
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Imagen de Entrada', fontsize=14, fontweight='bold', pad=10)
    
    # Añadir recuadro con predicción principal
    ax1.text(0.5, -0.05, f'Predicción: {pred_clase}\nConfianza: {confidence:.2f}%', 
             transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#4CAF50', alpha=0.8, edgecolor='black'),
             color='white')
    
    # 📊 DERECHA: Gráfico de barras de probabilidades
    colors = ['#4CAF50' if i == pred_idx else '#9b9b9b' for i in sorted_indices]
    bars = ax2.barh(range(len(sorted_probs)), sorted_probs * 100, color=colors, edgecolor='black', linewidth=0.5)
    
    # Configurar ejes
    ax2.set_yticks(range(len(sorted_classes)))
    ax2.set_yticklabels([c.replace('___', ' ').replace('__', ' ') for c in sorted_classes], fontsize=10)
    ax2.set_xlabel('Probabilidad (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(0, 100)
    
    # Añadir valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Invertir eje Y para que la mayor probabilidad esté arriba
    ax2.invert_yaxis()
    
    # Grid horizontal
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig,pred_idx




def obtener_recomendaciones(idx_):
    """
    Retorna las recomendaciones de una enfermedad formateadas para st.markdown().
    
    Parametro:
        idx_ (str o int): Indice de la enfermedad (0-18)
    
    Retorna:
        str: Texto en formato Markdown
    """
    
    idx_ = str(idx_)
    
    if idx_ not in enfermedades["enfermedades"]:
        return f"**Error:** No existe ninguna enfermedad con el indice '{idx_}'"
    
    info = enfermedades["enfermedades"][idx_]
    rec = info["recomendaciones"]
    
    lineas = []

    nombre = info['nombre']
    nombre_limpio = nombre.replace("_", " ")

    
    # Titulo
    lineas.append(f"#### {nombre_limpio}")
    lineas.append("")
    
    # Info basica
    lineas.append(f"**Cultivo:** {rec['cultivo']}")
    lineas.append(f"**Agente causal:** {info.get('agente_causal', 'No aplica')}")
    lineas.append("")
    
    # Descripcion
    lineas.append("#### Descripcion")
    lineas.append(rec['descripcion'])
    lineas.append("")
    
    # Medidas culturales o preventivas
    if "medidas_preventivas" in rec:
        lineas.append("#### Medidas Preventivas")
        medidas = rec["medidas_preventivas"]
    else:
        lineas.append("#### Medidas Culturales")
        medidas = rec.get("medidas_culturales", [])
    
    for medida in medidas:
        lineas.append(f"- {medida}")
    lineas.append("")
    
    # Manejo quimico
    lineas.append("#### Manejo Quimico")
    quimico = rec.get("manejo_quimico", [])
    
    if quimico:
        for tratamiento in quimico:
            lineas.append(f"- {tratamiento}")
    else:
        lineas.append("No aplica.")
    lineas.append("")
    
    # Monitoreo
    lineas.append("#### Monitoreo")
    lineas.append(rec.get('monitoreo', 'Revision periodica del cultivo.'))
    
    return "\n".join(lineas)
