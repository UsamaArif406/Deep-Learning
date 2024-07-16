import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

st.set_page_config(
    
    page_icon="🎲", 
    layout="wide",  # Set layout to wide for horizontal arrangement
)

st.title('Welcome to Handdigit classifier')

# Select mode: single or multiple image upload
mode = st.sidebar.selectbox('Select mode', ('Single Image', 'Multiple Images'))

if mode == 'Single Image':
    uploaded_file = st.sidebar.file_uploader('Upload a single image', type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded file as bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Decode the image bytes into an OpenCV image
        image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image", width=300)

        # Convert the OpenCV image to RGB format for TensorFlow
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to the target size
        image_resized = cv2.resize(image_rgb, (180, 180))
        # Convert the image to a NumPy array and expand dimensions to create a batch
        img_array = np.expand_dims(image_resized, axis=0)

        # Load the model
        model = tf.keras.models.load_model('my_model.hdf5')

        # Make predictions
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

        # Print the result
        st.write(
            f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence."
        )
    else:
        st.write("Please upload an image file to classify.")

elif mode == 'Multiple Images':
    uploaded_files = st.sidebar.file_uploader('Upload multiple images', type=["jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Read the uploaded file as bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # Decode the image bytes into an OpenCV image
            image = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            st.image(image, channels="BGR", caption=f"Uploaded Image: {uploaded_file.name}", width=300)

            # Convert the OpenCV image to RGB format for TensorFlow
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize the image to the target size
            image_resized = cv2.resize(image_rgb, (180, 180))
            # Convert the image to a NumPy array and expand dimensions to create a batch
            img_array = np.expand_dims(image_resized, axis=0)

            # Load the model
            model = tf.keras.models.load_model('my_model.hdf5')

            # Make predictions
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

            # Print the result
            st.write(
                f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence."
            )
    else:
        st.write("Please upload image files to classify.")
if st.sidebar.button('Training Results'):
    st.subheader('Visualizing Training History')
    path='Capture.PNG'
    st.image(path)