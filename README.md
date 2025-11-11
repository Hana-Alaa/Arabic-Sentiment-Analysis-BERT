# Arabic Sentiment Analysis 

**Arabic Sentiment Analyzer** is a web application built using **Streamlit** and **AraBERT** that classifies Arabic text into **positive** or **negative** sentiment.

---

## Features
- Predicts sentiment of Arabic text using fine-tuned AraBERT.
- Displays prediction label, confidence score, and visualization.
- Clean, RTL-friendly user interface built with Streamlit.
- Supports deployment via Streamlit Cloud or Hugging Face Spaces.

---

## 🧠 Model
The model is fine-tuned on an Arabic sentiment dataset using **AraBERTv2** from the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library.  

---

## Installation & Usage
1. Clone the repository:
```
git clone https://github.com/Hana-Alaa/Arabic-Sentiment-Analysis-BERT.git
cd Arabic-Sentiment-Analysis-BERT
```
2. Create a virtual environment:
```
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux / macOS
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the Streamlit app:
```
streamlit run app.py
```
📁 Project Structure
```
Arabic-Sentiment-Analysis-BERT/
│
├── app.py                    # Streamlit interface
├── src/
│   └──clean_text.py
│   └── data_prep.py
│   └── split_data.py 
│   ├── train_bert.py         # Model training script
│   └── test_bert.py          # Model evaluation script
├── models/                   # Model directory (excluded from Git)
├── requirements.txt
└── README.md
```
## Authors
- [Hana Alaa](https://github.com/Hana-Alaa)
- [Esraa Elfar](https://github.com/esraaelfar)
- [Manar Mohamed](https://github.com/manarmohamed45129-design)
