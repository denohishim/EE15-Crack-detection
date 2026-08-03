# GET324 - EE15: Concrete Bridge Deck Crack Detection (Cracked vs Non-Cracked)
# Streamlit app, following the same structure as Laboratory Exercise 10.
#
# Run locally:   streamlit run app.py
# (Model file expected at: models/mobilenetv3_transfer.keras)

# STEP 1: Import libraries for the Streamlit web application, data handling,
# and model loading
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# STEP 2: Configure the Streamlit web application's title, icon, and page layout
st.set_page_config(page_title="Bridge Deck Crack Detector", page_icon="🌉",
                    layout="centered")

# STEP 3: Load Saved Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenetv3_transfer.keras")
    return model

# STEP 4: Write the prediction function
def predict(model, pil_image):
    """Make prediction and return probabilities"""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    # class_names from training were ['cracked', 'non_cracked'] (alphabetical),
    # so the sigmoid output is P(non_cracked)
    prob_non_cracked = float(model.predict(arr, verbose=0)[0][0])
    prob_cracked = 1.0 - prob_non_cracked
    label = "Non-Cracked" if prob_non_cracked >= 0.5 else "Cracked"
    return label, prob_cracked * 100, prob_non_cracked * 100

# STEP 5: Building the User Interface (UI)
st.title("🌉 Concrete Bridge Deck Crack Detector")
st.write("Upload a photo of a concrete bridge deck surface to check for cracks.")

model = load_model()
uploaded_file = st.file_uploader("Upload a bridge deck image",
                                  type=["jpg", "jpeg", "png"])

# STEP 6: Make predictions and display the results
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    if st.button("Predict"):
        label, cracked_pct, non_cracked_pct = predict(model, img)
        st.write(f"**Prediction:** {label}")
        st.progress(int(cracked_pct), text=f"Cracked: {cracked_pct:.1f}%")
        st.progress(int(non_cracked_pct), text=f"Non-Cracked: {non_cracked_pct:.1f}%")
else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.caption("GET324 Mini-Project | Group EE15 | SDNET2018 Bridge Deck subset | MobileNetV3 transfer learning")

