import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model(r'BestModel_VGG_CNN_Tensorflow.h5')
class_names = ["Jahe", "Kunyit", "Lengkuas"]

def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[predicted_class], confidence

st.set_page_config(page_title="Klasifikasi Rempah", page_icon="🌿", layout="centered")
st.title("🌿 Klasifikasi Rempah")
st.write("Unggah gambar rempah seperti Jahe, Lengkuas, atau Kunyit, dan model kami akan mengidentifikasinya.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Prediksi Gambar"):
        with st.spinner("Sedang memproses gambar..."):
            label, confidence = predict_image(image)

        st.success("Prediksi Selesai!")
        st.write(f"**Label:** {label}")
        st.write(f"**Kepercayaan:** {confidence * 100:.2f}%")

st.markdown("---")
