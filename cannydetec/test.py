from matplotlib import pyplot as plt
import streamlit as st
from PIL import ImageTk, Image
import cv2
import numpy as np
st.set_page_config(
    layout="wide"
)

left_column, right_column = st.columns(2)
with left_column:
    first_img_container_left = st.container()
    second_img_container_left = st.container()

with right_column:
    mixer_component = st.container()
    result_img_container_right = st.container()

with first_img_container_left:
    left_column, right_column = st.columns(2)
    with left_column:
        uploaded_file1 = st.sidebar.file_uploader(
            "img1", type=['jpg', 'jpeg', 'png'])
        st.write("<p style='font-size:40px; text-align: center;'><b>Image 1</b></p>",
                 unsafe_allow_html=True)


    with right_column:
        option1 = st.selectbox("choose the desired image1 transformtion",('FT Magnitude', 'FT Phase', 'FT Real', 'FT Imaginary'))

with second_img_container_left:
    left_column, right_column = st.columns(2)
    with left_column:
        uploaded_file2 = st.sidebar.file_uploader(
            "img2", type=['jpg', 'jpeg', 'png'])
        st.write("<p style='font-size:40px; text-align: center;'><b>Image 2</b></p>",
                 unsafe_allow_html=True)


    with right_column:
        option2 = st.selectbox("choose the desired image2 transformtion",
                               ('FT Magnitude', 'FT Phase', 'FT Real', 'FT Imaginary'))

with mixer_component:
    output_option = st.selectbox("mixer output to", ('output1', 'output2'))
    left_column, right_column = st.columns(2)
    with left_column:
        comp1 = st.selectbox("component1", ('img1', 'img2'))
    with right_column:
        mixrate1 = st.slider("output1", 0, 100, 0)
    mix_comp1 = st.selectbox("Mag,phase,real,imaginary,uniphase,uniMag",
                             ('Mag', 'phase', 'real', 'imaginary', 'uniphase', 'uniMag'))
    left_column1, right_column1 = st.columns(2)
    with left_column1:
        comp2 = st.selectbox("component2", ('img1', 'img2'))
    with right_column1:
        mixrate2 = st.slider("output2", 0, 100, 0)
    mix_comp2 = st.selectbox("2Mag,phase,real,imaginary,uniphase,uniMag",
                             ('Mag', 'phase', 'real', 'imaginary', 'uniphase', 'uniMag'))

with result_img_container_right:
    left_column, right_column = st.columns(2)