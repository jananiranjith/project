# no chat history



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
from googletrans import Translator  # For translation

# ====== Setup ======
st.set_page_config(page_title="Tamil Translator Chatbot", page_icon="🗣️", layout="centered")

translator = Translator()

# ====== Load QA Dataset ======
@st.cache_resource
def load_qa_dataset(folder_path=r"C:\Users\User\OneDrive\Desktop\ai tamil bot\rag dataset"):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                file_data = json.load(f)
                for qa in file_data:
                    if "Q" in qa and "A" in qa:
                        data.append({"text": f"Q: {qa['Q']}\nA: {qa['A']}"} )
    return data

# ====== Load Models ======
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    documents = load_qa_dataset()
    texts = [doc["text"] for doc in documents]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return whisper_model, embed_model, index, documents

# ====== Load GGUF Model ======
@st.cache_resource
def load_custom_model():
    model = AutoModelForCausalLM.from_pretrained(
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
    return model

# ====== Load Resources ======
whisper_model, embed_model, faiss_index, documents = load_models()
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
    result = translator.translate(text, dest="ta")
    return result.text

def retrieve_top_k(query, k=3):
    query_embedding = embed_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), k)
    return "\n".join([documents[i]["text"] for i in I[0]])

def generate_answer(prompt):
    return model(prompt)

def speak_and_return_audio(text, lang="ta"):
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
st.title("🗣️ Tamil RAG Chatbot")

with st.expander("🎙️ Use Microphone Instead of Typing"):
    if st.button("🎤 Record Voice"):
        audio_path = record_audio()
        user_input = transcribe_audio(audio_path)
        st.success(f"🗣️ You said: `{user_input}`")
    else:
        user_input = ""

text_input = st.text_input("Type your question (or use mic above):", value=user_input)

if st.button("Send"):
    if text_input.strip():
        lang = detect_language(text_input)
        if lang == "en":
            text_input = translate_to_tamil(text_input)

        context = retrieve_top_k(text_input)
        prompt = f"""You are a helpful assistant that answers in Tamil.

{context}

User: {text_input}
Assistant:"""

        response = generate_answer(prompt)

        st.markdown(f"### 🤖 Response in Tamil:\n{response}")
        st.markdown(speak_and_return_audio(response), unsafe_allow_html=True)
    else:
        st.warning("Please enter or speak a question.")
