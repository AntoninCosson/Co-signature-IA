import os
import pandas as pd
from utils.text_loader import load_txt_files
from utils.reflexive_score import compute_reflexive_score
from utils.reflexive_graph import extract_concepts, build_reflexive_graph, compute_coherence_score
from utils.narrative_style import narrative_score
from utils.semantic_density import compute_semantic_density
from utils.gpt_shift import simulate_gpt_response, cosine_distance

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Chargement des textes
corpus_dir = "data"
fichiers = load_txt_files(corpus_dir)

results = []

for nom, texte in fichiers.items():
    print(f"\nTraitement de : {nom}")

    reflexive_score, _ = compute_reflexive_score(texte)
    concepts = extract_concepts(texte)
    G = build_reflexive_graph(concepts)
    coherence_score = compute_coherence_score(G)
    narrative, _ = narrative_score(texte)
    semantic_density = compute_semantic_density(texte)

    # GPT shift simulation
    question = "Quel est le rôle de la pensée dans la prise de décision ?"
    reponse_avant = simulate_gpt_response(question)
    reponse_apres = simulate_gpt_response(question + "\n\n" + texte)
    shift = cosine_distance(reponse_avant, reponse_apres)

    results.append({
        "fichier": nom,
        "reflexivite": reflexive_score,
        "coherence": coherence_score,
        "narrative": narrative,
        "semantic_density": semantic_density,
        "delta_vectoriel": shift
    })

# Export CSV
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
df.to_csv("results/rapport_scores.csv", index=False)
print("\n✅ Export CSV : results/rapport_scores.csv")

# Projection t-SNE
embeddings = df[["reflexivite", "coherence", "semantic_density", "delta_vectoriel", "narrative"]].values
labels = df["fichier"].values

perplexity = min(3, len(embeddings) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
reduced = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.title("Carte des styles de pensée réflexive")
colors = plt.cm.tab10.colors

for i, label in enumerate(labels):
    plt.scatter(reduced[i, 0], reduced[i, 1], color=colors[i % len(colors)])
    plt.text(reduced[i, 0]+5, reduced[i, 1], label, fontsize=8)

plt.tight_layout()
plt.savefig("results/projection_reflexive_tsne.png")
print("\n✅ Graphique exporté : results/projection_reflexive_tsne.png")