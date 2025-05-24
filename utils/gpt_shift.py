from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def simulate_gpt_response(prompt: str) -> str:
    if "pensée" in prompt:
        return (
            "La pensée est un processus cognitif qui permet à un individu de prendre des décisions, "
            "d'évaluer des options, et d'agir de manière consciente. Elle implique des mécanismes "
            "de raisonnement, de mémoire, et d’anticipation."
        )
    return "Ceci est une réponse générique simulée pour évaluer un changement d'orientation cognitive."

def cosine_distance(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(1 - sim, 4)