import os
import numpy as np
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.semantic_reflexive import semantic_reflexive_score

nlp = spacy.load("fr_core_news_md")


def simulate_gpt_response(question, priming_text=""):
    if priming_text:
        combined = priming_text + "\n" + question
    else:
        combined = question
    doc = nlp(combined)
    return doc.vector, combined


def gpt_shift_score(question, priming_text):
    base_vector, base_text = simulate_gpt_response(question)
    primed_vector, primed_text = simulate_gpt_response(question, priming_text)
    cos_sim = cosine_similarity([base_vector], [primed_vector])[0][0]
    shift = 1 - cos_sim
    return shift, cos_sim


if __name__ == "__main__":
    folder = "data/"
    question = "Quel est le rôle de la pensée dans la prise de décision ?"
    results = []

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            content = f.read()
        shift, cos = gpt_shift_score(question, content)
        results.append({
            "fichier": fname,
            "cosine": round(cos, 4),
            "delta_vectoriel": round(1 - cos, 4)
        })

    df = pd.DataFrame(results)
    df.to_csv("gpt_shift_results.csv", index=False)
    print("\nExport des shifts enregistré dans gpt_shift_results.csv")
    print(df.sort_values("delta_vectoriel", ascending=False).to_string(index=False))
