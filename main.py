import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Label names for the predictions
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 
    'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 
    'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 
    'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot', 
    'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# Sidebar for instructions
st.sidebar.title("🌿 Leaf Disease Detection")
st.sidebar.write("""
- Upload an image.
- The model will analyze and predict the disease type.
- Ensure the image is **clear and well-lit** for accurate results.
""")

# Load Model
st.sidebar.subheader("🧠 Model Information")
st.sidebar.write("✔ Model: **Deep Learning - Transfer Learning**")
st.sidebar.write("✔ Prediction Confidence: **Above 80% for reliability**")

st.title("🌱 Plant Disease Detection ")
st.write("Upload a image, and the AI model will detect possible diseases.")

# Load the trained model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# File Uploader
uploaded_file = st.file_uploader("📤 Upload a Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    img_resized = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150))
    normalized_image = np.expand_dims(img_resized, axis=0)

    # Display Uploaded Image
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    # Show progress bar while predicting
    with st.spinner("🔍 Analyzing... Please wait!"):
        predictions = model.predict(normalized_image)
    
    # Get the best prediction
    confidence = predictions[0][np.argmax(predictions)] * 100
    predicted_label = label_name[np.argmax(predictions)]
    
    # Display Prediction Results
    if confidence >= 80:
        st.success(f"✅ **Prediction:** {predicted_label}")
        st.write(f"📊 **Confidence Score:** {confidence:.2f}%")
    else:
        st.warning("⚠ Try uploading a different image. Confidence is too low.")
