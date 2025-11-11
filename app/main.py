import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")

# Try to load the pre-trained model if present. If not present, show a helpful
# Streamlit message with a link/instructions and keep `model` as None so the
# app doesn't crash when users try to classify images.
model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        # If model fails to load for some reason, surface the error in Streamlit
        # and continue so the app remains usable for debugging.
        st.error(f"Failed to load model at {model_path}: {e}")
        model = None
else:
    # Read a download link/file if available to help the user obtain the model
    link_file = os.path.join(working_dir, "trained_model", "trained_model_link.txt")
    download_link = None
    if os.path.exists(link_file):
        try:
            download_link = open(link_file, "r", encoding="utf-8").read().strip()
        except Exception:
            download_link = None

    st.warning("Pre-trained model not found at: ``" + model_path + "``")
    if download_link:
        st.markdown("**Download the trained model:**")
        st.write(download_link)
    else:
        st.info("Place your trained model file at: ``app/trained_model/plant_disease_prediction_model.h5``\n\nIf you don't have the model, check `app/trained_model/trained_model_link.txt` for a download URL.")

# loading the class names (safe load with error handling)
class_indices = {}
class_indices_path = os.path.join(working_dir, "class_indices.json")
if os.path.exists(class_indices_path):
    try:
        class_indices = json.load(open(class_indices_path, "r", encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load class indices from {class_indices_path}: {e}")
        class_indices = {}
else:
    st.error(f"class_indices.json not found at: {class_indices_path}")


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Ensure model is loaded before attempting prediction
            if model is None:
                st.error("Model is not available. Please download the trained model and place it at:\n``app/trained_model/plant_disease_prediction_model.h5``\nSee the link/README in `app/trained_model/trained_model_link.txt`.")
            else:
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
