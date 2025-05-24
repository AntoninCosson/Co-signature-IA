import os
import numpy as np
import pandas as pd
from utils.text_loader import load_txt_files
from utils.analysis import compute_shannon_entropy, compute_tfidf_top_terms, compute_average_syntax_depth
from utils.reflexive_score import compute_reflexive_score
from utils.reflexive_graph import extract_concepts, build_reflexive_graph, compute_coherence_score, embed_narrative, plot_tsne


if __name__ == "__main__":
    folder = "data/"
    texts = load_txt_files(folder)

    rows = []
    embeddings = []
    labels = []

    for fname, content in texts.items():
        print(f"\nTraitement de : {fname}")

        entropy = compute_shannon_entropy(content)
        tfidf = compute_tfidf_top_terms({fname: content})[fname][:3]  # top 3 termes
        depth = compute_average_syntax_depth(content)
        reflexive_score, _ = compute_reflexive_score(content)

        concepts = extract_concepts(content)
        G = build_reflexive_graph(concepts)
        coherence = compute_coherence_score(G)
        embedding = embed_narrative(G)

        rows.append({
            "fichier": fname,
            "entropie": entropy,
            "tfidf_1": tfidf[0][0] if tfidf else "",
            "tfidf_2": tfidf[1][0] if len(tfidf) > 1 else "",
            "tfidf_3": tfidf[2][0] if len(tfidf) > 2 else "",
            "syntaxe": depth,
            "reflexivite": reflexive_score,
            "coherence": coherence
        })

        embeddings.append(embedding)
        labels.append(fname)

    df = pd.DataFrame(rows)
    df.to_csv("rapport_scores.csv", index=False)
    print("\nExport CSV : rapport_scores.csv")

    if embeddings:
        print("\nProjection t-SNE...")
        plot_tsne(np.array(embeddings), labels)