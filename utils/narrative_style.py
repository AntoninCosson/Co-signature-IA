import re
from typing import List, Tuple

# Expressions stylistiques introspectives ou récursives
NARRATIVE_PATTERNS = [
    r"je me demande si",
    r"est-ce que je suis en train de",
    r"je crois que je pense",
    r"je suis en train de comprendre",
    r"il est possible que je sois",
    r"je réfléchis à la manière dont",
    r"je commence à entrevoir",
    r"comme si ma pensée",
    r"ce que je perçois",
    r"ce raisonnement me conduit à"
]

def detect_narrative_fragments(text: str) -> List[str]:
    """Détecte des fragments de style narratif introspectif."""
    sentences = re.split(r'[.!?]\\s+', text)
    results = []
    for sent in sentences:
        for pattern in NARRATIVE_PATTERNS:
            if re.search(pattern, sent, flags=re.IGNORECASE):
                results.append(sent.strip())
                break
    return results

def narrative_score(text: str) -> Tuple[float, List[str]]:
    """Calcule un score basé sur la densité de phrases introspectives."""
    sentences = re.split(r'[.!?]\\s+', text)
    matches = detect_narrative_fragments(text)
    score = len(matches) / max(len(sentences), 1)
    return round(score, 2), matches