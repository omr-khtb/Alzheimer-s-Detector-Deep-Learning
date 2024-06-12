import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

# Load the pre-trained Scratch Model
model = tf.keras.models.load_model('scratch.h5')

# Define the classes
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 299

def preprocess_image(image):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify_image(image, model):
    processed_image = preprocess_image(image)
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
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# Streamlit GUI
st.title("Image Classification with Scratch Model")
st.write("Upload an image to classify it using the pre-trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save("uploaded_image.png")  # Save the image to a file
    st.markdown(
        """
        <img src='uploaded_image.png' class='animated-image' width='700' />
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.write("Classifying...")

    predictions = classify_image(image, model)
    predicted_class = CLASSES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    if predicted_class != 'NonDemented':
        gif_url = "https://media.giphy.com/media/l0HlOvJ7yaacpuSas/giphy.gif"
        st.markdown(
            f"""
            <script>
            alert('Sorry, the patient is diagnosed with Alzheimer\'s disease.');
            setTimeout(function() {{
                var gif = document.createElement('img');
                gif.src = "{gif_url}";
                gif.style.width = "100%";
                document.body.appendChild(gif);
            }}, 100);
            setTimeout(function() {{
                document.body.removeChild(document.body.lastChild);
            }}, 5000);
            </script>
            """,
            unsafe_allow_html=True
        )
    else:
        gif_url = "https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif"
        st.markdown(
            f"""
            <script>
            alert('Good news! The patient does not have Alzheimer\'s disease.');
            setTimeout(function() {{
                var gif = document.createElement('img');
                gif.src = "{gif_url}";
                gif.style.width = "100%";
                document.body.appendChild(gif);
            }}, 100);
            setTimeout(function() {{
                document.body.removeChild(document.body.lastChild);
            }}, 5000);
            </script>
            """,
            unsafe_allow_html=True
        )
