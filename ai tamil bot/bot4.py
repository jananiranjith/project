# only answers related to dtaaset , otherwiesse from llama



import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np
import os
import json
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
from gtts import gTTS
from ctransformers import AutoModelForCausalLM
from googletrans import Translator

# ====== Setup ======
st.set_page_config(page_title="ILAMAI AI Chatbot", page_icon="📚", layout="centered")
st.title("ILAMAI AI Chatbot")
translator = Translator()

# ====== Load QA Datasets ======
@st.cache_resource
def load_qa_data(base_folder):
    category_data = {}
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            qa_list = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        for qa in file_data:
                            if "Q" in qa and "A" in qa:
                                qa_list.append(qa)
            category_data[folder.lower()] = qa_list
    return category_data

# ====== Load Models and Indexes ======
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load datasets
    data = load_qa_data("rag dataset")
    indexes = {}

    for category, qa_pairs in data.items():
        questions = [q["Q"] for q in qa_pairs]
        embeddings = embed_model.encode(questions, show_progress_bar=True)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings))
        indexes[category] = {"index": index, "data": qa_pairs}

    return whisper_model, embed_model, indexes

@st.cache_resource
def load_custom_model():
    return AutoModelForCausalLM.from_pretrained(
        "abhinand/tamil-llama-7b-instruct-v0.2-GGUF",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>"],
        local_files_only=False
    )

whisper_model, embed_model, indexes = load_models()
model = load_custom_model()

# ====== Utilities ======
def record_audio(file_path="input.wav", duration=5, fs=16000):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(file_path, fs, audio)
    return file_path

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_tamil(text):
    return translator.translate(text, dest="ta").text

def retrieve_answer(query, threshold=0.5):
    query_embedding = embed_model.encode([query])
    for category, info in indexes.items():
        index, data = info["index"], info["data"]
        D, I = index.search(np.array(query_embedding), k=1)
        if D[0][0] < threshold:
            return data[I[0][0]]["A"]
    return None

def get_rag_context(query, k=3):
    general_data = indexes.get("conversational", {}).get("data", [])
    general_index = indexes.get("conversational", {}).get("index")
    if not general_data or not general_index:
        return ""
    query_embedding = embed_model.encode([query])
    D, I = general_index.search(np.array(query_embedding), k)
    return "\n".join([f"Q: {general_data[i]['Q']}\nA: {general_data[i]['A']}" for i in I[0]])

def generate_answer(prompt):
    return model(prompt)

def speak_response(text, lang="ta"):
    if not text.strip():
        return "❌ No response to convert to speech."
    try:
        tts = gTTS(text=text, lang=lang)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        return f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        return f"❌ TTS failed: {str(e)}"

# ====== UI ======
with st.expander("🎧 Use Microphone Instead of Typing"):
    if st.button("🎤 Record Voice"):
        audio_path = record_audio()
        user_input = transcribe_audio(audio_path)
        st.success(f"🎙️ You said: `{user_input}`")
    else:
        user_input = ""

text_input = st.text_input("Type your question (or use mic above):", value=user_input)

if st.button("Send"):
    if text_input.strip():
        if detect_language(text_input) == "en":
            text_input = translate_to_tamil(text_input)

        # Try dataset retrieval first
        response = retrieve_answer(text_input)
        if not response:
            context = get_rag_context(text_input)
            prompt = f"""You are a helpful assistant. Always respond in Tamil.

{context}

User: {text_input}
Assistant:"""
            response = generate_answer(prompt)

        st.markdown(f"### 🤖 Response in Tamil:\n{response}")
        st.markdown(speak_response(response), unsafe_allow_html=True)
    else:
        st.warning("Please enter or speak a question.")
