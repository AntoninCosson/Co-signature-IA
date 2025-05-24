import pandas as pd

def compute_transformation_score(row):
    # Pondération simple — à affiner si besoin
    return round(
        0.3 * row['reflexivite'] +
        0.3 * row['coherence'] +
        0.2 * row['semantic_density'] +
        0.2 * row['delta_vectoriel'], 4
    )

if __name__ == "__main__":
    # Chargement des scores intermédiaires
    reflexive_data = pd.read_csv("results/rapport_scores.csv")
    shift_data = pd.read_csv("results/gpt_shift_results.csv")

    # Jointure
    df = pd.merge(reflexive_data, shift_data, on="fichier")

    # Placeholder densité sémantique si non présente
    if 'semantic_density' not in df.columns:
        df['semantic_density'] = 0.5  # Par défaut neutre

    # Calcul du score final
    df['transformation_score'] = df.apply(compute_transformation_score, axis=1)

    # Export
    df.to_csv("results/transformation_scores_final.csv", index=False)
    print("✅ Exporté : results/transformation_scores_final.csv")