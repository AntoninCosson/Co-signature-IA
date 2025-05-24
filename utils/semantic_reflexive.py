import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("fr_core_news_md")

# Prototype de phrases réflexives (à affiner avec ton corpus)
REFLEXIVE_PROTOTYPES = [
    "Je me demande si ce que je perçois est réel ou simulé.",
    "Ce processus cognitif semble émerger d'une boucle de rétroaction.",
    "La conscience peut être vue comme une structure logique en mouvement.",
    "Je me base sur des données internes pour construire une représentation mentale."
]

# Pré-calculer les vecteurs des prototypes
prototype_vectors = [nlp(p).vector for p in REFLEXIVE_PROTOTYPES]


def semantic_reflexive_score(sentence):
    """
    Calcule la similarité cosinus entre une phrase et les prototypes réflexifs.
    Retourne le score moyen (0.0 - 1.0).
    """
    doc = nlp(sentence)
    if not doc.has_vector:
        return 0.0

    vec = doc.vector.reshape(1, -1)
    sims = cosine_similarity(vec, prototype_vectors)
    return float(np.mean(sims))


# Exemple d'utilisation
if __name__ == "__main__":
    phrase = "Je crois que ma conscience se structure en fonction du langage."
    score = semantic_reflexive_score(phrase)
    print(f"Score sémantique réflexif : {score:.3f}")
