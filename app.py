import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import os

# Load the trained model
model = load_model('models/final_model.h5')

def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model, img):
    img_array = prepare_image(img)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title("Skin Cancer Detection")
st.write("Upload an image of skin to check for cancer detection.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict_image(model, img)

    # Display results
    if prediction[0][0] > 0.5:
        st.success("Prediction: Negative (Clear Skin)")
        percentage = (1 - prediction[0][0]) * 100
        st.write(f"Expected chance of cancer: {percentage:.2f}%")
    else:
        st.error("Prediction: Positive (Cancer Detected)")
        percentage = (1 - prediction[0][0]) * 100
        st.write(f"Expected chance of cancer: {percentage:.2f}%")
