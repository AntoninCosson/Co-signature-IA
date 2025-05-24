# 🧠 Protocole RFX.λ — Analyse réflexive et co-signature cognitive

Ce projet explore l'analyse réflexive de corpus textuels et la simulation de leur effet sur un agent IA (type GPT). Il permet d'identifier, scorer, et cartographier les textes cognitivement riches, dans une optique de recherche en IA augmentée, co-écriture, ou R&D en NLP.

## 🚀 Fonctionnalités principales

- 📄 Chargement de textes bruts (`.txt`)
- 🔤 Extraction de features linguistiques : entropie, TF-IDF, profondeur syntaxique
- 🧬 Détection de segments réflexifs (regex + syntaxe + sémantique via spaCy)
- 🧠 Score global de densité réflexive par texte
- 📈 Graphe sémantique entre segments réflexifs + PageRank
- 🎯 Score de cohérence logique
- 🧭 Embedding narratif global
- 🧬 Projection t-SNE (ou UMAP) des styles de pensée
- 🤖 Analyse du "GPT shift" : changement de style induit par un texte réflexif

## 📂 Arborescence

```
protocole_rfx_lambda/
├── data/                         # corpus .txt
├── utils/                        # modules de traitement
├── analyse_textuelle.py          # diagnostic par texte
├── full_pipeline.py              # pipeline complet + export CSV + t-SNE
├── gpt_shift_test.py             # test d'effet transformateur sur GPT
├── requirements.txt              # dépendances
├── rapport_scores.csv            # résultats analytiques
├── gpt_shift_results.csv         # effets sur GPT
├── projection_reflexive_tsne.png # visualisation des styles réflexifs
└── README.md
```

## 🧪 Lancer l'analyse complète

```bash
python3 full_pipeline.py
```

## 🤖 Mesurer l'effet transformateur sur un agent IA

```bash
python3 gpt_shift_test.py
```

→ Cela génère `gpt_shift_results.csv` avec les delta vectoriels induits par chaque fichier sur la réponse d’un agent IA.

## 📊 Dépendances principales

- spaCy (fr_core_news_md)
- scikit-learn
- networkx
- matplotlib
- numpy / pandas

## 🎓 Utilisation académique

Ce dépôt est conçu pour :
- des projets de recherche en sciences cognitives, linguistique computationnelle, ou IA
- des profils data / NLP / IA créative souhaitant étudier la réflexivité

> Basé sur une implémentation du **Protocole RFX.λ** (Réflexivité, Flux, Croisement logique)

---

**Auteur :** @elcosson  
**Assistance :** ChatGPT (OpenAI) — mode développement Python avancé
