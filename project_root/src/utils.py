import re
import nltk

def ensure_nltk_packages():
    """Ensure required NLTK packages are downloaded."""
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def get_actual_metrics(text):
    """
    Calculates actual linguistic bins (sentences and nouns) from generated text.
    
    Args:
        text (str): The text with or without control tokens.
        
    Returns:
        tuple: (s_bin, n_bin, density_value)
    """
    ensure_nltk_packages()
    
    # Remove control tokens to get pure text
    clean = re.sub(r'<\|.*?\|>', '', text).strip()
    
    # Calculate sentences
    sentences = [s for s in re.split(r'[.!?]+', clean) if len(s.strip()) > 3]
    
    # Calculate Noun Density
    words = nltk.word_tokenize(clean)
    tags = nltk.pos_tag(words)
    nouns = len([w for w, t in tags if t.startswith('NN')])
    density = (nouns / len(words)) * 100 if words else 0
    
    # Binning logic mapped from research papers
    # Sentences: Bin 1 (<=2), Bin 2 (<=5), Bin 3 (>5)
    s_bin = 1 if len(sentences) <= 2 else (2 if len(sentences) <= 5 else 3)
    
    # Nouns Density: Bin 1 (<=5%), Bin 2 (<=15%), Bin 3 (>15%)
    n_bin = 1 if density <= 5 else (2 if density <= 15 else 3)
    
    return s_bin, n_bin, round(density, 2)

def is_successful(attr, actual_val, target_bin):
    """
    Checks if the generated text matches the target bin.
    """
    if attr == "sents":
        return 1 if actual_val == target_bin else 0
    if attr == "NOUN":
        # Based on noun density thresholds
        if target_bin == 1: return 1 if actual_val <= 5 else 0
        if target_bin == 2: return 1 if 5 < actual_val <= 15 else 0
        if target_bin == 3: return 1 if actual_val > 15 else 0
    return 0
