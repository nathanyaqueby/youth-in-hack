import streamlit as st
from PIL import Image
from fastai.vision.all import *


st.title("Hello World")

# load the fastai model
export_path = pathlib.Path('export.pkl')
learn = load_learner(export_path)

# create a file uploader
uploader = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

# create a button
if st.button("Predict"):
    # read the image
    img = Image.create(uploader)
    # make a prediction
    pred, pred_idx, probs = learn.predict(img)
    # display the prediction
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
    # display the image
    st.image(img.to_thumb(500,500), caption='Uploaded Image')
