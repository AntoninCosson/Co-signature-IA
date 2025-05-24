from utils.reflexive_segments import detect_reflexive_segments
from utils.spaCy_reflexive import structural_reflexive_score
from utils.semantic_reflexive import semantic_reflexive_score
import re


def compute_reflexive_score(text, weight_regex=0.3, weight_syntax=0.3, weight_semantic=0.4):
    """
    Calcule un score global de densité réflexive en combinant trois sources :
    - regex : détection de segments via expressions régulières
    - syntaxe : détection de structures introspectives avec spaCy
    - sémantique : similarité avec prototypes réflexifs
    """
    regex_segments = detect_reflexive_segments(text, min_matches=1)
    regex_score = len(regex_segments) / max(len(re.split(r"[.!?]", text)), 1)

    syntax_score = structural_reflexive_score(text)

    sentences = re.split(r"[.!?]", text)
    semantic_scores = [semantic_reflexive_score(s) for s in sentences if s.strip()]
    semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0

    score = weight_regex * regex_score + weight_syntax * syntax_score + weight_semantic * semantic_score
    return score, regex_segments