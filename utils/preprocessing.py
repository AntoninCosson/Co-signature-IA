# utils/preprocessing.py
import re
import spacy

nlp = spacy.load("fr_core_news_md")

def clean_text(text):
    """Nettoie un texte brut (espaces, ponctuation, casse)."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

def tokenize_text(text):
    """Tokenisation spaCy avec filtres simples."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]