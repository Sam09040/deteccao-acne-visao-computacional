import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Detecção de Acne", layout="wide")

st.title("Detecção de Acne com YOLO")
st.markdown("Envie uma imagem facial para detectar e visualizar as lesões de acne.")

@st.cache_resource
def load_model(model_path: str = "models/weights/best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado em: {model_path}")
        st.stop()
    model = YOLO(model_path)
    return model

model = load_model()

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem original")
        st.image(image, use_container_width=True)

    with st.spinner("Processando inferência..."):
        results = model.predict(source=np.array(image), conf=0.25, save=False)

    result_img = results[0].plot()
    result_img_rgb = result_img[..., ::-1]

    with col2:
        st.subheader("Detecções")
        st.image(result_img_rgb, use_container_width=True)

    # Informações das detecções
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Resumo das inferências")
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        for cls_name, count in class_counts.items():
            st.write(f"- **{cls_name}**: {count} detecção(ões)")
        st.write(f"**Total:** {len(boxes)} objeto(s) detectado(s)")
    else:
        st.info("Nenhuma lesão de acne detectada com o limiar de confiança atual.")
else:
    st.info("Aguardando envio de imagem...")
