#generall from llama , 10th from 10th

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

# ====================== CONFIG ======================
st.set_page_config(page_title="ILAMAI AI Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
    .main { padding: 1rem; }
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("📘 ILAMAI AI Chatbot")
translator = Translator()

# ====================== Load Models ======================
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embed_model

@st.cache_resource
def load_llama_model():
    return AutoModelForCausalLM.from_pretrained(
        "abhinand/tamil-llama-7b-instruct-v0.2-GGUF",
        model_type="llama",
        max_new_tokens=1024,  # Increased for longer answers
        temperature=0.7,
        top_k=40,
        top_p=0.9,
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

def get_10th_answer(query, threshold=0.7):  # increased threshold for more relevance
    q_embed = embed_model.encode([query])
    D, I = index_10th.search(np.array(q_embed), k=1)
    if D[0][0] < threshold:
        return answers_10th[I[0][0]]
    return None

def get_llama_answer(query):
    prompt = f"""
நீங்கள் ஒரு பயனுள்ள உதவியாளர். கீழ்க்காணும் கேள்விக்கு விரிவாக தமிழில் பதிலளியுங்கள்:

User: {query}
Assistant:
"""
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
col1, col2 = st.columns(2)

with col1:
    st.subheader("💬 Enter your question")
    mode = st.radio("Choose Mode", ["General", "Education (10th)"])
    user_input = st.text_area("Type your question here:")
    if st.button("Send") and user_input.strip():
        user_query = user_input.strip()
        if detect_language(user_query) == "en":
            user_query = translate_to_tamil(user_query)

        if mode == "Education (10th)":
            answer = get_10th_answer(user_query)
            if not answer:
                st.session_state.response = "Couldn't find an answer in 10th dataset."
            else:
                st.session_state.response = answer
        else:
            st.session_state.response = get_llama_answer(user_query)

with col2:
    st.subheader("🤖 Answer")
    if "response" in st.session_state:
        st.markdown(st.session_state.response)
        st.markdown(speak_text(st.session_state.response), unsafe_allow_html=True)
