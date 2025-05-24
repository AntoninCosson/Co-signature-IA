# utils/analysis.py

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

def compute_shannon_entropy(text):
    """Calcule l'entropie de Shannon d'un texte (sur les caractères)."""
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    total = len(text)
    entropy = -sum((count/total) * math.log2(count/total) for count in freq.values())
    return entropy

def compute_tfidf_top_terms(texts_dict, top_n=10):
    """
    Calcule les top termes TF-IDF pour chaque texte.
    :param texts_dict: dict {filename: content}
    :return: dict {filename: [(term, score), ...]}
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    docs = list(texts_dict.values())
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    results = {}
    for idx, (fname, _) in enumerate(texts_dict.items()):
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        results[fname] = top_terms
    return results

nlp = spacy.load("fr_core_news_md")

def compute_average_syntax_depth(text):
    """
    Calcule la profondeur syntaxique moyenne d’un texte à partir des arbres de dépendance spaCy.
    """
    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            depths.append(depth)
    return sum(depths) / len(depths) if depths else 0.0