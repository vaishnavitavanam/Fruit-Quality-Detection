
    
import os
# Setting the legacy Keras environment variable for compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 2. THIS MUST BE THE VERY FIRST STREAMLIT COMMAND (STRICT RULE)
st.set_page_config(page_title="Fruit Quality Detector", page_icon="🍎")

# 3. Background Image Function
def set_bg_hack():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                         url("https://images.unsplash.com/photo-1610832958506-aa56368176cf?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
             background-size: cover;
             background-position: center;
             background-attachment: fixed;
         }}
         /* Force text to be readable */
         h1, h2, h3, h4, p, span, label, .stMarkdown {{
             color: white !important;
         }}
         .stFileUploader {{
             background-color: rgba(255, 255, 255, 0.1);
             padding: 20px;
             border-radius: 15px;
             backdrop-filter: blur(10px);
         }}
         /* 3. Uploader Box - Solid White making it stand out */
             .stFileUploader {{
                 background-color: rgba(255, 255, 255, 0.95) !important; /* Almost solid white */
                 padding: 30px;
                 border-radius: 15px;
                 border: 2px solid #4CAF50; /* Optional: Green border for fruit theme */
             }}

             /* 4. Fix for "Drag and Drop" and "Limit" text to BLACK */
             .stFileUploader section div div {{
                 color: #000000 !important;
                 font-weight: 500;
             }}
             
             /* 5. Fix for "Browse Files" button text to BLACK */
             .stFileUploader button p {{
                 color: #000000 !important;
             }}

             /* 6. Fix for Uploaded file name/info to BLACK */
             .stFileUploader div[data-testid="stFileUploaderFileData"] p {{
                 color: #000000 !important;
             }}
            
             
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack()
# Page Configuration
#st.set_page_config(page_title="Fruit Quality Detector", page_icon="🍎")
st.title("🍎 Fruit Quality Detection")
st.subheader("Using Deep Learning for Rotten and Fresh Fruits Classification") 
st.write("Upload a fruit image to check its quality (Fresh or Rotten)")

@st.cache_resource
def load_fruit_model():
    """
    Loads the pre-trained Keras model from the local directory.
    Uses @st.cache_resource to load the model only once.
    """
    try:
        # If the model is in the same folder as fr.py, using the filename is sufficient
        model_path = "fruit_quality_model.h5" 
        
        # Loading the model without compilation to avoid version-specific optimizer errors
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize the model
model = load_fruit_model()

# File Uploader UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Open image and convert to RGB (standardizes PNG/JPG formats)
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # 2. Verify model was loaded successfully before attempting prediction
    if model is not None:
        with st.spinner('Analyzing image quality...'):
            # Image Preprocessing
            # Resize image to match the input size used during training (224x224)
            img_resized = img.resize((224, 224))
            
            # Convert image to array and normalize pixel values to [0, 1]
            img_array = image.img_to_array(img_resized) / 255.0
            
            # Expand dimensions to create a batch of 1
            img_array = np.expand_dims(img_array, axis=0)

            # Class Labels - These MUST match the order of your training subfolders
            classes = [
                'Freshapple', 'Freshbanana', 'Freshgrapes', 'Freshmango', 'Freshorange', 
                 'Rottenbanana', 'Rottengrapes', 'Rottenmango', 'Rottenorange','rottenapple',
            ]

            # Perform Prediction
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100

            # UI Results Display
            st.divider()
            st.write("### Prediction Results:")
            
            predicted_class = classes[class_index]
            
            # Display success message for Fresh fruits and error for Rotten ones
            if "Fresh" in predicted_class:
                st.success(f"**Classification:** {predicted_class}")
            else:
                st.error(f"**Classification:** {predicted_class}")
                
            st.info(f"**Confidence Level:** {confidence:.2f}%")
    else:
        st.error("The model could not be loaded. Please ensure 'fruit_quality_model.h5' is in the project directory.")
