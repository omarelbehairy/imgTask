"""import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from PIL import ImageTk, Image
import cv2

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

        logger.info('Uploading image 1...')

    with right_column:
        option1 = st.selectbox("choose the desired image1 transformtion",('FT Magnitude', 'FT Phase', 'FT Real', 'FT Imaginary'))

with second_img_container_left:
    left_column, right_column = st.columns(2)
    with left_column:
        uploaded_file2 = st.sidebar.file_uploader(
            "img2", type=['jpg', 'jpeg', 'png'])
        st.write("<p style='font-size:40px; text-align: center;'><b>Image 2</b></p>",
                 unsafe_allow_html=True)

        logger.info('Uploading image 2...')

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
"""

from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logging.debug('Received request to upload image')
    return render_template('Htmlfile.html')

@app.route('/upload', methods=['POST'])
def upload():
    print('Received request to upload image')

    if 'file' not in request.files:
        print('No file uploaded')
        return 'No file uploaded', 400

    file = request.files['file']
    print(f'Received file: {file.filename}')


    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        # Save the uploaded file to the uploads directory
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Read the uploaded image using OpenCV
        img = cv2.imread(filename)

        # Process the image (e.g., apply some filters)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the processed image temporarily
        processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
        cv2.imwrite(processed_filename, gray)

        # Return the processed image file
        return send_file(processed_filename, mimetype='image/png')

    return 'File format not supported', 400

if __name__ == '__main__':
    app.run(debug=True)
