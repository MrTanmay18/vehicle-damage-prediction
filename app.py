import streamlit as st
from model_helper import predict


st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload a file",type=["jpg","jpeg","png"])

if uploaded_file:
    image_path="tem_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption="Uploaded File",use_container_width=True)
        prediction = predict(image_path)
        st.info(f"predicted class: {prediction}")