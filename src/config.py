SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
TOXIC_MODEL = "unitary/toxic-bert"
NER_MODEL = "dslim/bert-base-NER"
TOPIC_MODEL = "facebook/bart-large-mnli"
SUMM_MODEL = "facebook/bart-large-cnn"
LANG_MODEL = "papluca/xlm-roberta-base-language-detection"

INTRO_TEXT = """
# NLP Studio: Multi Tool AI

Paste text and explore sentiment, emotion, toxicity, topics, entities, summaries, and language detection in one place.
"""

POS_WORDS = {
    "love", "great", "amazing", "excellent", "happy", "wonderful",
    "fantastic", "awesome", "perfect", "smooth"
}

NEG_WORDS = {
    "hate", "bad", "terrible", "awful", "horrible", "worst",
    "broken", "buggy", "slow", "disappointing", "refund"
}
