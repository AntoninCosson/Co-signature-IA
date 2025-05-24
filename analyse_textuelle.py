# analyse_textuelle.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_loader import load_txt_files
from utils.analysis import compute_tfidf_top_terms, compute_shannon_entropy, compute_average_syntax_depth
from utils.reflexive_score import compute_reflexive_score

if __name__ == "__main__":
    folder = "data/"
    texts = load_txt_files(folder)
    print(f"{len(texts)} fichiers chargés.")

    for fname, content in texts.items():
        print(f"\n--- {fname} ---")

        entropy = compute_shannon_entropy(content)
        print(f"Entropie de Shannon : {entropy:.4f}")

        top_terms = compute_tfidf_top_terms({fname: content})
        print("Top 10 TF-IDF :")
        for term, score in top_terms[fname]:
            print(f"  {term}: {score:.4f}")

        depth = compute_average_syntax_depth(content)
        print(f"Profondeur syntaxique moyenne : {depth:.2f}")

        reflexive_score, segments = compute_reflexive_score(content)
        print(f"Score de densité réflexive (fusionné) : {reflexive_score:.2f}")
        print("Segments à forte densité réflexive :")
        for seg in segments:
            print(f"  → {seg[:80]}...")