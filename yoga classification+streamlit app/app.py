import streamlit as st
import cv2 
import numpy as np
from keras.models import load_model
from PIL import Image

model=load_model('yoga_posses_final.h5')

class_labels = ['Downdog Pose', 'Goddess Pose', 'Plank Pose', 'Tree Pose', 'Warrior Pose']

# Function to perform image classification
def classify_image(image):
    resized_image = cv2.resize(image, (150, 150))
    scaled_image = resized_image / 255.0
    reshaped_img = scaled_image.reshape(1, 150, 150, 3)
    prediction = np.argmax(model.predict(reshaped_img))
    predicted_class = class_labels[prediction]
    return predicted_class

# Streamlit app
def main():
    option=st.sidebar.radio('option',('Single File','Multiple Files'))
    if option=='Single File':
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg",'png'], accept_multiple_files=False)
        if uploaded_file is not None:
                image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
                st.image(image, caption='Uploaded Image',width=200)
    elif option == "Multiple Files":
        uploaded_files = st.sidebar.file_uploader("Choose multiple images...", type=["jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

                st.image(image, caption='Uploaded Image', width=200)
    if st.sidebar.button('Predict'):
        # Perform classification
        predicted_class = classify_image(image)
        st.write("Predicted Pose:", predicted_class)

if __name__ == '__main__':
    main()