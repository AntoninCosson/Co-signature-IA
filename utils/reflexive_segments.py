import re

# Groupes de motifs réflexifs enrichis
REFLEXIVE_PATTERNS = {
    "cognition": [
        r"\bje (me )?demande\b",
        r"\bon dirait que\b",
        r"\bje crois que\b",
        r"\bce que je perçois\b"
    ],
    "logique": [
        r"\ble mécanisme de\b",
        r"\bla structure du\b",
        r"\bce processus\b",
        r"\bcela implique\b"
    ],
    "style_ia": [
        r"\bsimulation (cognitive|symbolique)\b",
        r"\bschéma informationnel\b",
        r"\bprocessus émergent\b",
        r"\barchitecture réflexive\b"
    ]
}

def detect_reflexive_segments(text, min_matches=1):
    sentences = re.split(r"[.!?]", text)
    results = []
    for sent in sentences:
        count = sum(
            bool(re.search(pat, sent, re.IGNORECASE))
            for group in REFLEXIVE_PATTERNS.values()
            for pat in group
        )
        if count >= min_matches:
            results.append(sent.strip())
    return results

def reflexive_score(segments, text):
    """Score réflexif = proportion de phrases réflexives dans le texte."""
    total_sentences = len(re.split(r"[.!?]", text))
    if total_sentences == 0:
        return 0.0
    return len(segments) / total_sentences