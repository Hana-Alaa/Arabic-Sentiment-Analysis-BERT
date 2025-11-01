# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import base64
import os, base64, mimetypes, streamlit.components.v1 as components

# ---------------- Config ----------------
MODEL_DIR = "models/arabertv2-sentiment"
MAX_LENGTH = 128
# POSITIVE_GIF = "images/positive.png"
# NEGATIVE_GIF = "images/negative.png"
# ----------------------------------------

# 1. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    low_cpu_mem_usage=False
)
model.eval()
model.to("cpu")

# 2. Streamlit UI setup
st.set_page_config(page_title="Arabic Sentiment Analyzer", page_icon="📝", layout="centered")
st.markdown("""
    <style>
        img {
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 10px !important;
            max-width: 200px !important;
        }
    </style>

    <h1 style='text-align: center; color: #fff;'>
        📝 Arabic Sentiment Analysis
    </h1>
    <p style='text-align: center; font-size:18px; color: #aaa;'>
        Discover the mood of Arabic texts with ease and clarity
    </p>
""", unsafe_allow_html=True)

# 3. Arabic text input box with RTL and placeholder
st.markdown("""
    <style>
        textarea {
            direction: rtl;
            font-size: 16px;
            height: 150px;
        }
        div.stButton > button {
            width: 50%;
            font-size: 18px;
            padding: 10px 0;
            display: block;
            margin: 10px auto;
            border-color: #fff;
            color: #fff;
        }
        .st-emotion-cache-ocqkz7 {
            display: flex;
            justify-content: center; 
            align-items: center;      
        }
    </style>
""", unsafe_allow_html=True)

user_input = st.text_area("", placeholder="ادخل النص هنا...")

# 4. Prediction button
predict_button = st.button("Predict")

# 5. Prediction logic
if predict_button:
    if user_input.strip() == "":
        st.warning("من فضلك أدخل نص لتتمكن من التنبؤ!")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze()
            pred_label = torch.argmax(probs).item()

        # Display results
        label_map = {0: "Negative", 1: "Positive"}
        colors = {0: "red", 1: "green"}
        messages = {
            0: " النص يحمل مشاعر سلبية",
            1: " النص إيجابي ومليء بالطاقة الجميلة"
        }

        # Calculate Percentage
        percent = int(probs[pred_label].item() * 100)

        #  Label + Percentage
        st.markdown(
            f"<h2 style='text-align:center; color:{colors[pred_label]};'>"
            f"{label_map[pred_label]} ({percent}%)</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center; font-size:18px;'>{messages[pred_label]}</p>",
            unsafe_allow_html=True
        )
        # gif_path = POSITIVE_GIF if pred_label == 1 else NEGATIVE_GIF
        # st.markdown(
        #     f"<div style='text-align:center; margin-top:20px;'>"
        #     f"<img src='{gif_path}' width='400'>"
        #     f"</div>",
        #     unsafe_allow_html=True
        # )

        if pred_label == 1:
            emoji = "😄"
        else:
            emoji = "😞"

        st.markdown(
            f"""
            <div style="text-align:center; font-size:80px; margin-top:10px;">
                {emoji}
            </div>
            """,
            unsafe_allow_html=True
        )

        # st.markdown(
        #     f"<p style='text-align:center; font-size:16px;'>"
        #     f"Negative: {int(probs[0].item()*100)}% &nbsp;&nbsp;&nbsp; "
        #     f"Positive: {int(probs[1].item()*100)}%</p>",
        #     unsafe_allow_html=True