import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from datetime import datetime
import base64
from pymongo import MongoClient
import bcrypt
from bson.objectid import ObjectId

# --------------------------
# CSS for aesthetics
# --------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(160deg, #0f2027, #203a43, #2c5364); color: #f5f5f5 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
.title-hero { text-align: center; font-size: 3rem; color: #18a36d; font-weight: 900; margin-bottom:0.3rem; margin-top:0.1rem; }
.subtitle-hero { text-align: center; font-size: 1.3rem; color: #ffffff; margin-bottom: 1.3rem; font-weight:600; }
.chat-user { background-color:#DCF8C6; color:#000000; padding:10px; border-radius:12px; margin:5px 0; max-width:70%; margin-left:auto; }
.chat-bot { background-color:#ECECEC; color:#000000; padding:10px; border-radius:12px; margin:5px 0; max-width:70%; margin-right:auto; }
.chat-timestamp { font-size:0.75rem; color:#555; display:block; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Cloud Configuration Setup
# --------------------------
try:
    MONGO_URI = st.secrets["MONGO_URI"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # This keeps your code from crashing locally but doesn't leak secrets to GitHub
    MONGO_URI = "" 
    GROQ_API_KEY = ""

client = MongoClient(MONGO_URI)
db = client['plantdb']
user_collection = db['users']
plant_collection = db['userplants']

# --------------------------
# Password hashing helpers
# --------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

# --------------------------
# Authentication
# --------------------------
if "user" not in st.session_state:
    st.sidebar.title("Welcome")
    option = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])

    if option == "Sign Up":
        st.sidebar.subheader("Create new account")
        new_user = st.sidebar.text_input("Username", key="signup_user")
        new_password = st.sidebar.text_input("Password", type="password", key="signup_pass")
        if st.sidebar.button("Sign Up"):
            if user_collection.find_one({"username": new_user}):
                st.sidebar.error("Username already exists!")
            elif not new_user or not new_password:
                st.sidebar.error("Username and password cannot be empty.")
            else:
                hashed_pw = hash_password(new_password)
                user_collection.insert_one({"username": new_user, "password": hashed_pw})
                st.session_state["user"] = new_user
                st.rerun()
    else:
        st.sidebar.subheader("Login to your account")
        username = st.sidebar.text_input("Username", key="login_user")
        password = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            user = user_collection.find_one({"username": username})
            if user and verify_password(password, user['password']):
                st.session_state["user"] = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
    st.stop()

user_id = st.session_state["user"]

if st.sidebar.button("Logout"):
    st.session_state.pop("user")
    st.rerun()

# --------------------------
# Load Models & LLM
# --------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

@st.cache_resource
def load_mobilenet_model():
    return tf.keras.models.load_model("mobilenet_plantvillage.h5")

@st.cache_resource
def load_class_names():
    return np.load("class_names.npy", allow_pickle=True)

yolo_model = load_yolo_model()
mobilenet_model = load_mobilenet_model()
class_names = load_class_names()

# Use Groq for Cloud LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# --------------------------
# App Header
# --------------------------
st.markdown(f'<div class="title-hero">🌱 Smart Plant Chatbot - {user_id}</div>', unsafe_allow_html=True)

# --------------------------
# Plants Dashboard
# --------------------------
st.markdown("## 🌿 My Plants")
plants = list(plant_collection.find({"user_id": user_id}))

col1, col2 = st.columns([3, 1])
with col1:
    plant_options = {f"{p['disease']} ({p['confidence']:.1f}%)": str(p['_id']) for p in plants}
    if plant_options:
        selected_label = st.selectbox("Select a plant", list(plant_options.keys()))
        selected_id = plant_options[selected_label]
        selected_plant = next(p for p in plants if str(p['_id']) == selected_id)
    else:
        selected_plant = None
        st.info("No saved plants.")
with col2:
    if selected_plant and st.button("Delete Selected"):
        plant_collection.delete_one({"_id": ObjectId(selected_id)})
        st.rerun()

# --------------------------
# Add New Plant
# --------------------------
st.markdown("## ➕ Add a New Plant")
input_type = st.radio("Input type:", ["Upload Image", "Use Webcam"], horizontal=True)

def detect_and_classify(img_bgr):
    results = yolo_model.predict(img_bgr, conf=0.3, verbose=False)
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0: return None, None, img_bgr
    x1,y1,x2,y2 = map(int, boxes[0].xyxy[0])
    cropped = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (224,224))
    arr = np.expand_dims(keras_image.img_to_array(resized) / 255.0, axis=0)
    preds = mobilenet_model.predict(arr)
    disease, conf = class_names[np.argmax(preds)], float(np.max(preds) * 100)
    cv2.rectangle(img_bgr, (x1,y1),(x2,y2),(24,163,109),3)
    return disease, conf, img_bgr

img_bgr = None
if input_type == "Upload Image":
    file = st.file_uploader("Upload leaf image:", type=["jpg","jpeg","png"])
    if file: img_bgr = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
else:
    cam_image = st.camera_input("Take photo")
    if cam_image: img_bgr = cv2.cvtColor(np.array(Image.open(cam_image)), cv2.COLOR_RGB2BGR)

if img_bgr is not None:
    predicted_disease, confidence, processed_img = detect_and_classify(img_bgr)
    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), width=400)
    if predicted_disease and st.button("Save Plant"):
        _, img_encoded = cv2.imencode('.jpg', processed_img)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        plant_collection.insert_one({"user_id": user_id, "disease": predicted_disease, "confidence": confidence, "image_b64": img_b64, "timestamp": datetime.now(), "chat_history": []})
        st.success("Saved!")
        st.rerun()

# --------------------------
# Chat Section (WITH ALL LANGUAGES)
# --------------------------
if selected_plant:
    st.markdown(f"### 💬 Chat about: {selected_plant['disease']}")
    
    indian_languages = {
        "English": "en", "Hindi (हिन्दी)": "hi", "Gujarati (ગુજરાતી)": "gu",
        "Marathi (મરાठी)": "mr", "Bengali (বাংলা)": "bn", "Telugu (తెలుగు)": "te",
        "Tamil (தமிழ்)": "ta", "Kannada (ಕನ್ನಡ)": "kn", "Malayalam (മലയാളം)": "ml",
        "Punjabi (ਪੰਜਾਬੀ)": "pa", "Urdu (اردو)": "ur", "Odia (ଓଡ଼ିଆ)": "or", "Assamese (অসমীয়া)": "as"
    }
    lang_choice = st.selectbox("🌐 Language:", list(indian_languages.keys()))
    lang = indian_languages[lang_choice]

    if selected_plant.get("chat_history"):
        for chat in selected_plant["chat_history"]:
            st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🤖 <b>Bot:</b> {chat["answer"]}</div>', unsafe_allow_html=True)

    user_q = st.text_input("Ask about this disease...")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            q_en = GoogleTranslator(source=lang, target='en').translate(user_q) if lang != "en" else user_q
            prompt = PromptTemplate.from_template("Answer this in simple terms for plant disease '{disease}': {question}")
            
            # Groq Call
            res = llm.invoke(prompt.format(disease=selected_plant['disease'], question=q_en))
            full_ans_en = res.content
            
            # Summary
            sum_res = llm.invoke(f"Summarize this in 2 lines: {full_ans_en}")
            short_ans_en = sum_res.content
            
            # Translate back
            ans = GoogleTranslator(source="en", target=lang).translate(short_ans_en) if lang != "en" else short_ans_en
            
            plant_collection.update_one({"_id": selected_plant['_id']}, {"$push": {"chat_history": {"question": user_q, "answer": ans, "timestamp": datetime.now()}}})
            st.rerun()