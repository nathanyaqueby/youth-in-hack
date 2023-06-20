import streamlit as st
from PIL import Image#
import pathlib
from fastai.vision.all import *


st.title("Hello World")

# load the fastai model
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
export_path = pathlib.Path('export.pkl')
learn = load_learner(export_path)

# create a file uploader
uploader = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

# create a button
if st.button("Predict"):
    # read the image
    img = PILImage.create(uploader)
    # predict the image
    pred, pred_idx, probs = learn.predict(img)
    # display the image
    st.image(img)
    # display the prediction
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
