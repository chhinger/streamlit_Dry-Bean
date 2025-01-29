import streamlit as st
import torch
import tensorflow as tf
import numpy as np

# Load Models
@st.cache_resource
def load_pytorch_model():
    model = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))  # Example model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model

@st.cache_resource
def load_tensorflow_model():
    return tf.keras.models.load_model("mlp_model.h5")

# Title
st.title("Dry Bean Classification App")

# Model Selection
model_choice = st.selectbox("Select Model", ["PyTorch", "TensorFlow"])

# User Input
st.write("Enter feature values:")
features = [st.number_input(f"Feature {i+1}") for i in range(4)]

# Predict Button
if st.button("Predict"):
    features_array = np.array([features], dtype=np.float32)

    if model_choice == "PyTorch":
        model = load_pytorch_model()
        input_tensor = torch.tensor(features_array)
        prediction = model(input_tensor).detach().numpy()
    else:
        model = load_tensorflow_model()
        prediction = model.predict(features_array)

    # Display Result
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

