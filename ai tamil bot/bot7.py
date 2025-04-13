# Enhanced ILAMAI AI Chatbot with UI Improvements and Tanglish-to-Tamil
import streamlit as st
import os
import json
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
import faiss
from gtts import gTTS
from io import BytesIO
import base64
from ctransformers import AutoModelForCausalLM
from googletrans import Translator
from PIL import Image
from pathlib import Path

# ====================== CONFIG ======================
st.set_page_config(page_title="ILAMAI AI Chatbot", page_icon="🤖", layout="wide")

background_image_path = Path(r"C:\Users\User\OneDrive\Desktop\ai tamil bot\dark theme bg.png").as_posix()

st.markdown(rf"""
    <style>
    .main {{ padding: 1rem; }}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem;}}

    .stApp {{
        background-image: 
                          url("data:image/png;base64,{base64.b64encode(open(background_image_path, "rb").read()).decode()}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}

    @import url('https://fonts.googleapis.com/css2?family=Baloo+Thambi+2&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Baloo Thambi 2', sans-serif;
        color: white !important;
    }}

    textarea, .stTextInput, .stRadio label, .stButton > button {{
        color: white !important;
    }}

    .question-container {{
        background-color: rgba(0,0,0,0.4);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }}

    .answer-container {{
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        color: black;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("📘 ILAMAI AI Chatbot")
translator = Translator()

# ====================== Load Models ======================
@st.cache_resource
def load_models():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_llama_model():
    return AutoModelForCausalLM.from_pretrained(
    "abhinand/tamil-llama-7b-instruct-v0.2-GGUF",
    model_type="llama",
    max_new_tokens=1024,
    temperature=0.0,  # <<< Make deterministic
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.1,
    stop=["</s>"]
)


embed_model = load_models()
llama_model = load_llama_model()

# ====================== Load Dataset ======================
@st.cache_resource
def load_10th_dataset(folder_path="10th dataset"):
    questions, answers = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    questions.append(item["Q"])
                    answers.append(item["A"])
    embeddings = embed_model.encode(questions, show_progress_bar=True)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return questions, answers, index

questions_10th, answers_10th, index_10th = load_10th_dataset()

# ====================== Helpers ======================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "ta"

def translate_to_tamil(text):
    return translator.translate(text, dest="ta").text

def get_10th_answer(query, threshold=0.7):
    q_embed = embed_model.encode([query])
    D, I = index_10th.search(np.array(q_embed), k=1)
    if D[0][0] < threshold:
        return answers_10th[I[0][0]]
    return None

def get_llama_answer(query):
    prompt = f"""கீழ்காணும் கேள்விக்கு தமிழில் மட்டும் பதிலளிக்கவும். விளக்கம் தேவையில்லை.

கேள்வி: {query}
பதில்:"""
    return llama_model(prompt)

def speak_text(text):
    try:
        tts = gTTS(text=text, lang="ta")
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        return f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        return f"TTS Error: {e}"

# ====================== Layout ======================
st.markdown("""---""")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color:white;'>💬 உங்கள் கேள்வியை உள்ளிடுங்கள்</h3>", unsafe_allow_html=True)

    mode = st.radio("முறை தேர்ந்தெடுக்கவும்", ["General", "Education (10th)"], key="mode_radio")
    user_input = st.text_area("இங்கே உங்கள் கேள்வியை டைப் செய்யவும்:", key="user_question")

    if st.button("✉️ Send") and user_input.strip():
        user_query = user_input.strip()

        lang = detect_language(user_query)
        if lang == "en":
            user_query = translate_to_tamil(user_query)

        st.session_state.question = user_query

        if mode == "Education (10th)":
            answer = get_10th_answer(user_query)
            st.session_state.response = answer or "10வது தர வினாக்கள் தொகுப்பில் பதில் காணவில்லை."
        else:
            st.session_state.response = get_llama_answer(user_query)

with col2:
    if "question" in st.session_state:
        st.markdown(f"""
        <div class='question-container'>
            <b>📩 கேள்வி:</b><br>{st.session_state.question}
        </div>
        """, unsafe_allow_html=True)

    if "response" in st.session_state:
        st.markdown(f"""
        <div class='answer-container'>
            <b>🤖 பதில்:</b><br>{st.session_state.response}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(speak_text(st.session_state.response), unsafe_allow_html=True)

# ====================== Footer ======================
st.markdown("""
<hr style='margin-top:50px;'>
<div style='text-align:center; color:white;'>
    <small>🚀 Made with ❤️ | Powered by ILAMAI AI</small>
</div>
""", unsafe_allow_html=True)
