import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Roboflow API details
API_KEY = "qrk4LOkABlkyTUP8Fem8"  # Replace with your actual Roboflow API key
MODEL_ID = "pest-detection-yolov8"
MODEL_VERSION = "3"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}/{MODEL_VERSION}?api_key={API_KEY}"

# Pest-specific organic solutions
organic_solutions = {
    "ants": "Use cinnamon, vinegar, or lemon juice near entry points. Plant mint or cloves to repel ants.",
    "bees": "Avoid chemical sprays. Relocate using a professional beekeeper. Plant bee-friendly flowers away from high-traffic areas.",
    "beetles": "Introduce natural predators like ladybugs. Use neem oil or diatomaceous earth around plants.",
    "caterpillars": "Spray neem oil or garlic-pepper spray. Introduce natural predators like birds or parasitic wasps.",
    "grasshoppers": "Spray garlic solution on plants. Introduce birds or chickens in the area.",
    "moth": "Use pheromone traps. Spray a mixture of neem oil and water on plants.",
    "slugs": "Place crushed eggshells or coffee grounds around plants. Use beer traps to attract and remove them.",
    "wasps": "Hang fake wasp nests. Spray peppermint oil solution near nests.",
    "weevils": "Store grains in airtight containers. Freeze infected grains for 72 hours.",
    "earthworms": "Mix a tablespoon of mustard powder with water and pour it over the area—this drives earthworms to the surface so you can relocate them.",
    "snails":"try using crushed eggshells or coffee grounds around your plants as barriers—snails dislike rough surfaces.",
    "earwigs":"Fill a shallow container with vegetable oil and a bit of soy sauce, then place it near affected areas; earwigs are drawn to the scent, fall in, and are trapped. Another option is to lay rolled-up damp newspapers near your plants overnight—earwigs love hiding in them. In the morning, simply pick up the newspaper and dispose of it."
}

# Streamlit UI
st.title("🐛 Pest Detection & Organic Solutions Web App")
st.write("Upload an image or use your webcam for real-time pest detection.")

# Choose input method
option = st.radio("Choose an input method:", ("📸 Use Webcam", "📂 Upload Image"))

def detect_pests(image):
    """Sends image to Roboflow API and processes results."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    response = requests.post(API_URL, files={"file": buffered.getvalue()})

    if response.status_code == 200:
        result = response.json()
        detected_pests = set()

        if "predictions" in result and len(result["predictions"]) > 0:
            for detection in result["predictions"]:
                class_name = detection["class"].lower()
                confidence = detection["confidence"]
                detected_pests.add(class_name)
                st.write(f"✅ **Detected Pest:** {class_name.capitalize()} (Confidence: {confidence:.2f})")

            st.image(image, caption="Processed Image", use_column_width=True)

            for pest in detected_pests:
                if pest in organic_solutions:
                    st.subheader(f"🌱 Organic Solution for {pest.capitalize()}:")
                    st.write(organic_solutions[pest])
                else:
                    st.write(f"🚧 No organic solution found for {pest.capitalize()}.")
        else:
            st.warning("🚫 No pest detected in the image.")
    else:
        st.error("❌ Error in detection. Please check API key or model settings.")

# Webcam input
if option == "📸 Use Webcam":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        frame_placeholder = st.empty()
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            if st.button("📷 Capture & Detect"):
                st.write("Processing image...")
                image = Image.fromarray(frame)
                detect_pests(image)

        cap.release()

# File upload input
elif option == "📂 Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        detect_pests(image)
