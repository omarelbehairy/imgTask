from matplotlib import pyplot as plt
import streamlit as st
from PIL import ImageTk, Image
import cv2
import numpy as np
st.set_page_config(
    layout="wide"
)

left_column , middle_column , right_column = st.columns(3)
with left_column:
    first_img_container_left = st.container()

with middle_column:
    second_img_container_right = st.container()

with right_column:
    tools_container = st.container()

with first_img_container_left:
    uploaded_image = st.sidebar.file_uploader(
        "img1", type=['jpg', 'jpeg', 'png'])

with second_img_container_right:
    filtered_image = st.container()

with tools_container:
    user_input = st.number_input("Enter sigma value", 0, 10000, 1)
    user_input = st.number_input("Enter kernel size", 0, 10000, 1)
    left_column, right_column = st.columns(2)
    with left_column:
        comp1 = st.slider("lower threshold",50, 150, 50)
    with right_column:
        mixrate1 = st.slider("upper threshold", 80, 200, 80)
    

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #show the image on the web
    first_img_container_left.image(uploaded_image, use_column_width=False)
    second_img_container_right.image(Image.fromarray(gray), use_column_width=False)