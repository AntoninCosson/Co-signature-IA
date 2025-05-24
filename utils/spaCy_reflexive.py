import spacy

nlp = spacy.load("fr_core_news_md")

# Liste indicative de verbes introspectifs
INTROSPECTIVE_VERBS = {"penser", "croire", "supposer", "réfléchir", "imaginer", "comprendre", "percevoir"}


def is_reflexive_structure(sent):
    """
    Détecte si une phrase contient un sujet personnel + verbe introspectif + compléments.
    Cible les structures comme "je pense que...", "nous croyons que..."
    """
    for token in sent:
        if token.pos_ == "VERB" and token.lemma_ in INTROSPECTIVE_VERBS:
            subj = [w for w in token.children if w.dep_ in ("nsubj", "nsubj:pass")]
            if any(s.text.lower() in ("je", "nous") for s in subj):
                return True
    return False


def structural_reflexive_score(text):
    """
    Retourne un score entre 0 et 1 = % de phrases contenant une structure réflexive.
    """
    doc = nlp(text)
    if not doc.sents:
        return 0.0
    total = 0
    reflexive_count = 0
    for sent in doc.sents:
        total += 1
        if is_reflexive_structure(sent):
            reflexive_count += 1
    return reflexive_count / total if total > 0 else 0.0