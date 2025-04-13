# Enhanced ILAMAI AI Chatbot with auto PDF download for model papers
import streamlit as st
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from io import BytesIO
import base64
from ctransformers import AutoModelForCausalLM
from pathlib import Path
from PIL import Image

# ====================== CONFIG ======================
st.set_page_config(page_title="ILAMAI AI Chatbot", page_icon="🤖", layout="wide")

background_image_path = Path(r"C:\Users\User\OneDrive\Desktop\ai tamil bot\dark theme bg.png").as_posix()
with open(background_image_path, "rb") as f:
    encoded_bg = base64.b64encode(f.read()).decode()

st.markdown(rf"""
    <style>
    .main {{ padding: 1rem; }}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem;}}
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_bg}");
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
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.1,
        stop=["</s>"]
    )

embed_model = load_models()
llama_model = load_llama_model()

# ====================== Load 10th Dataset ======================
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
def get_10th_answer(query, threshold=0.7):
    q_embed = embed_model.encode([query])
    D, I = index_10th.search(np.array(q_embed), k=1)
    if D[0][0] < threshold:
        return answers_10th[I[0][0]]
    return None

def get_llama_answer(query):
    prompt = f"""கீழ்காணும் கேள்விக்கு தமிழில் மட்டும் பதிலளிக்கவும். விளக்கம் தேவையில்லை.\n\nகேள்வி: {query}\nபதில்:"""
    return llama_model(prompt)

# ====================== Layout ======================
st.markdown("""---""")
col1, col2 = st.columns(2)

# Model question papers mapping (PDF paths)
pdf_papers = {
    "10th Board Model Question Paper": r"C:\Users\User\OneDrive\Desktop\ai tamil bot\10th-Tamil-Public-Exam-April-2025-Original-Question-Paper-PDF-Download.pdf",
    "10th Quarterly Question Paper": r"C:\Users\User\OneDrive\Desktop\ai tamil bot\10th-Tamil-Quarterly-Exam-2024-Question-Paper-with-Answer-Keys-Perambalur-District-PDF-Download.pdf",
    "10th Half-Yearly Question Paper": r"C:\Users\User\OneDrive\Desktop\ai tamil bot\10Th TAMIL TVR Half Yearly Question Paper 2024.pdf"
}

with col1:
    st.markdown("<h3 style='color:white;'>💬 உங்கள் கேள்வியை உள்ளிடுங்கள்</h3>", unsafe_allow_html=True)

    mode = st.radio("முறை தேர்ந்தெடுக்கவும்", ["General", "Education (10th)", "Model Question Paper"])

    if mode == "Model Question Paper":
        paper_type = st.selectbox("🔹 கேள்வி தாள் வகையைத் தேர்ந்தெடுக்கவும்", list(pdf_papers.keys()))
        selected_pdf_path = pdf_papers.get(paper_type)
        if selected_pdf_path:
            st.success("✅ தேர்ந்தெடுத்த கேள்வித்தாள் தயார்!")
            with open(selected_pdf_path, "rb") as f:
                st.download_button(
                    label="📥 இந்த கேள்வி தாளை பதிவிறக்கவும்",
                    data=f,
                    file_name=os.path.basename(selected_pdf_path),
                    mime="application/pdf"
                )
    else:
        user_input = st.text_area("இங்கே உங்கள் கேள்வியை டைப் செய்யவும்:", key="user_question")

        if st.button("✉️ Send") and user_input.strip():
            query = user_input.strip()
            st.session_state.question = query

            if mode == "Education (10th)":
                answer = get_10th_answer(query)
                st.session_state.response = answer or "10வது தர வினாக்கள் தொகுப்பில் பதில் காணவில்லை."
            else:
                st.session_state.response = get_llama_answer(query)

with col2:
    if "question" in st.session_state:
        st.markdown(f"""
        <div class='question-container'>
            <b>📩 கேள்வி:</b><br>{st.session_state.question}
        </div>
        """, unsafe_allow_html=True)

    if "response" in st.session_state and mode != "Model Question Paper":
        st.markdown(f"""
        <div class='answer-container'>
            <b>🤖 பதில்:</b><br>{st.session_state.response}
        </div>
        """, unsafe_allow_html=True)

# ====================== Footer ======================
st.markdown("""
<hr style='margin-top:50px;'>
<div style='text-align:center; color:white;'>
    <small>🚀 Made with ❤️ | Powered by ILAMAI AI</small>
</div>
""", unsafe_allow_html=True)
