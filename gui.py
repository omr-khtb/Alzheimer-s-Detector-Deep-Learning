import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D
from PIL import Image
import numpy as np
import base64
import io

# Ensure that the custom objects are registered
custom_objects = {'SeparableConv2D': SeparableConv2D}

# Load the pre-trained models with error handling
def load_model(model_path, custom_objects=None):
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

scratch_model = load_model('scratch.h5', custom_objects=custom_objects)
xception_model = load_model('Xception.h5', custom_objects=custom_objects)
mobilenet_model = load_model('mobilenet.h5', custom_objects=custom_objects)
cnn_model = load_model('cnn.h5', custom_objects=custom_objects)

# Define the classes
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
SCRATCH_IMG_SIZE = 299
XCEPTION_IMG_SIZE = 150
MOBILENET_IMG_SIZE = 224  # Assuming MobileNet uses 224x224 input size
CNN_IMG_SIZE = 299  # Adjust based on your CNN model's input size

def preprocess_image(image, img_size):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify_image(image, model, img_size):
    processed_image = preprocess_image(image, img_size)
    predictions = model.predict(processed_image)
    return predictions

# Add CSS for background image and animation
def add_custom_css():
    with open("bak.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        @keyframes slide-in-rotate {{
            0% {{
                transform: translateX(100%) rotate(0deg);
                opacity: 0;
            }}
            100% {{
                transform: translateX(0) rotate(360deg);
                opacity: 1;
            }}
        }}
        .animated-image {{
            animation: slide-in-rotate 2s ease-in-out;
            width: 400px;
        }}
        .stAlert p {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# Streamlit GUI
st.title("Do you have Alzheimer's? Find Out with Image Classification")
st.write("Upload an image to classify it using the selected pre-trained model.")

model_choice = st.selectbox("Choose a model", ("Scratch Model", "Xception Model", "MobileNet Model", "CNN Model"))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Display the image with animation
    st.markdown(
        f"""
        <img src="data:image/png;base64,{img_str}" class="animated-image" />
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    st.write("Classifying...")

    if model_choice == "Scratch Model" and scratch_model is not None:
        predictions = classify_image(image, scratch_model, SCRATCH_IMG_SIZE)
    elif model_choice == "Xception Model" and xception_model is not None:
        predictions = classify_image(image, xception_model, XCEPTION_IMG_SIZE)
    elif model_choice == "MobileNet Model" and mobilenet_model is not None:
        predictions = classify_image(image, mobilenet_model, MOBILENET_IMG_SIZE)
    elif model_choice == "CNN Model" and cnn_model is not None:
        predictions = classify_image(image, cnn_model, CNN_IMG_SIZE)
    else:
        st.error("Selected model is not available.")
        predictions = None

    if predictions is not None:
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

        if predicted_class != 'NonDemented':
            st.error("Sorry, the patient is diagnosed with Alzheimer's disease.")
            gif_url = "https://media.tenor.com/l3BF03rFfh0AAAAM/bestcry-sad.gif"
        else:
            st.success("Good news! The patient does not have Alzheimer's disease.")
            gif_url = "https://media.tenor.com/LX846cKh4nUAAAAM/happy-emotional.gif"

        st.markdown(
            f"""
            <img src="{gif_url}" style="width:300px; display:block; margin:auto;" />
            """,
            unsafe_allow_html=True
        )

# To run this Streamlit app, save it to a file (e.g., app.py) and use the command:
# streamlit run app.py
