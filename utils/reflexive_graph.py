# reflexive_graph.py

import spacy
import networkx as nx
import numpy as np
from itertools import combinations
from utils.semantic_reflexive import semantic_reflexive_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

nlp = spacy.load("fr_core_news_md")


def extract_concepts(text, threshold=0.65):
    """
    Extrait les phrases avec un score sémantique réflexif significatif.
    Retourne une liste de (index, phrase, vector, score)
    """
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    result = []
    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        if not doc.has_vector:
            continue
        vec = doc.vector
        score = semantic_reflexive_score(sent)
        if score >= threshold:
            result.append((i, sent, vec, score))
    return result


def build_reflexive_graph(concepts, similarity_threshold=0.70):
    """
    Construit un graphe orienté des segments réflexifs.
    Noeuds = phrases, Arcs = proximité sémantique.
    """
    G = nx.DiGraph()
    for idx, sent, vec, score in concepts:
        G.add_node(idx, text=sent, score=score, vector=vec)

    for (i1, s1, v1, _), (i2, s2, v2, _) in combinations(concepts, 2):
        sim = cosine_similarity([v1], [v2])[0][0]
        if sim >= similarity_threshold:
            G.add_edge(i1, i2, weight=sim)
    return G


def compute_coherence_score(G):
    """
    Mesure de la cohérence logique globale : moyenne des similarités sémantiques sur les arcs.
    """
    if G.number_of_edges() == 0:
        return 0.0
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    return sum(weights) / len(weights)


def embed_narrative(G):
    """
    Calcule un embedding global du texte par moyenne pondérée des vecteurs des nœuds.
    """
    if not G.nodes:
        return np.zeros((nlp.vocab.vectors_length,))
    vectors = np.array([d['vector'] for _, d in G.nodes(data=True)])
    return np.mean(vectors, axis=0)


def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
    plt.title("Carte des styles de pensée réflexive")
    plt.savefig("projection_reflexive_tsne.png", dpi=300)
    plt.show()


def print_graph_summary(G):
    print(f"\nGraphe : {len(G.nodes)} noeuds, {len(G.edges)} connexions")
    centrality = nx.pagerank(G)
    top_nodes = sorted(centrality.items(), key=lambda x: -x[1])[:3]
    print("\nTop 3 concepts centraux :")
    for idx, score in top_nodes:
        print(f"  • {G.nodes[idx]['text'][:80]}... ({score:.3f})")


if __name__ == "__main__":
    import os
    folder = "data/"
    embeddings = []
    labels = []

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            text = f.read()
        concepts = extract_concepts(text)
        G = build_reflexive_graph(concepts)
        print(f"\n--- {fname} ---")
        print_graph_summary(G)
        coherence = compute_coherence_score(G)
        print(f"Score de cohérence réflexive : {coherence:.2f}")
        emb = embed_narrative(G)
        embeddings.append(emb)
        labels.append(fname)

    if embeddings:
        plot_tsne(embeddings, labels)
