# ⚡ NLP Studio — Multi-Tool AI

This project is a simple web application that lets you run several NLP tools on the same text in one place.

It uses:

- Sentiment analysis  
- Emotion detection  
- Toxicity detection  
- Named entity recognition  
- Zero shot topic classification  
- Summarization  
- Language detection  

All models are from Hugging Face and the interface is built with Gradio.

## Models

- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `j-hartmann/emotion-english-distilroberta-base`
- `unitary/toxic-bert`
- `dslim/bert-base-NER`
- `facebook/bart-large-mnli`
- `facebook/bart-large-cnn`
- `papluca/xlm-roberta-base-language-detection`

## How to run

```bash
pip install -r requirements.txt
python app.py
