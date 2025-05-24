import spacy
from collections import Counter
from typing import Tuple

nlp = spacy.load("fr_core_news_md")  # Assure-toi que ce modèle est installé

def compute_semantic_density(text: str) -> float:
    """
    Calcule une densité sémantique basée sur la diversité lexicale significative.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    total = len(tokens)
    unique = len(set(tokens))
    density = unique / total if total > 0 else 0
    return round(density, 3)

if __name__ == "__main__":
    with open("data/exemple.txt", "r", encoding="utf-8") as f:
        texte = f.read()
    print("Densité sémantique :", compute_semantic_density(texte))