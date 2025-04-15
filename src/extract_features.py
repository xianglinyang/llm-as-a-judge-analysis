import spacy
import re
import numpy as np # Optional, if you want a NumPy array output
from collections import Counter
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math
import argparse
import json
from tqdm import tqdm
from typing import Tuple, List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the spaCy model (use a consistent model for all texts)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Define the order and names of features for the output vector
SURFACE_FEATURE_NAMES = [
    "char_count",
    "token_count",
    "sentence_count",
    "paragraph_count",
    "avg_token_length",
    "punctuation_ratio",
    "bullet_point_lines",
    "numbered_list_lines",
    "all_caps_word_ratio",
    "avg_paragraph_length_tokens",
    "avg_paragraph_length_sents",
]

def extract_surface_formatting_features_vector(text: str) -> list:
    """
    Extracts surface and formatting features from text and returns them as a vector (list).

    Args:
        text: The input text string.

    Returns:
        A list containing the calculated feature values in a predefined order,
        or None if the input text is empty/invalid.
    """
    if not text or not text.strip():
        # Return a vector of zeros or None if input is invalid,
        # ensure consistency based on how you handle missing data later.
        # Returning zeros might be simpler for direct use in ML models.
        return [0.0] * len(SURFACE_FEATURE_NAMES)
        # return None

    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    sentences = list(doc.sents)
    # Split paragraphs by double newline and filter empty ones
    paragraphs_text = [p.strip() for p in text.split('\n\n') if p.strip()]

    # --- Basic Counts ---
    char_count = float(len(text))
    token_count = float(len(tokens))
    sentence_count = float(len(sentences))
    paragraph_count = float(len(paragraphs_text))

    # Denominators for normalization (avoid division by zero)
    safe_token_count = token_count if token_count > 0 else 1.0
    safe_paragraph_count = paragraph_count if paragraph_count > 0 else 1.0
    safe_char_count = char_count if char_count > 0 else 1.0

    # --- Calculate Features ---
    avg_token_length = sum(len(t.text) for t in tokens) / safe_token_count
    punctuation_ratio = sum(1 for t in tokens if t.is_punct) / safe_token_count
    bullet_point_lines = float(len(re.findall(r'^\s*[\*\-]\s+', text, re.MULTILINE)))
    numbered_list_lines = float(len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE)))
    all_caps_word_ratio = sum(1 for t in tokens if t.text.isupper() and len(t.text) > 1) / safe_token_count

    # Avg paragraph length requires reprocessing paragraph text
    avg_paragraph_length_tokens = sum(len([t for t in nlp(p) if not t.is_space]) for p in paragraphs_text) / safe_paragraph_count
    avg_paragraph_length_sents = sum(len(list(nlp(p).sents)) for p in paragraphs_text) / safe_paragraph_count

    # --- Create Feature Vector in Predefined Order ---
    feature_vector = [
        char_count,
        token_count,
        sentence_count,
        paragraph_count,
        avg_token_length,
        punctuation_ratio,
        bullet_point_lines,
        numbered_list_lines,
        all_caps_word_ratio,
        avg_paragraph_length_tokens,
        avg_paragraph_length_sents,
    ]

    return feature_vector


# Define lists for specific word categories (customize these)
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "consequently",
    "nevertheless", "thus", "hence", "otherwise", "indeed", "additionally",
    "subsequently", "finally", "similarly", "likewise"
    # Add more as needed
}

HEDGE_WORDS = {
    "might", "could", "may", "perhaps", "possibly", "suggests", "appears",
    "seems", "likely", "probably", "often", "sometimes", "generally",
    "assume", "estimate", "tend", "believe", "feel" # Added some weaker ones
    # Add more as needed
}
# Simple contraction list (approximation) - can be improved with regex
CONTRACTIONS_SUFFIXES = {"n't", "'re", "'ve", "'ll", "'s", "'m", "'d"}

# Define the order and names of features for the output vector
LEXICAL_FEATURE_NAMES = [
    "ttr",                     # Type-Token Ratio (surface forms)
    "lemma_ttr",               # Type-Token Ratio (lemmas)
    "transition_word_ratio",   # Frequency of transition words
    "hedge_word_ratio",        # Frequency of hedge words
    "contraction_ratio",       # Approximate frequency of contractions
    # Add slots for POS ratios here if considered lexical (can be debated)
    # e.g., "pos_NOUN_ratio", "pos_VERB_ratio", etc. (calculated in the main function)
]
# We'll dynamically add POS ratio names later if needed

def extract_lexical_features_vector(text: str, include_pos_ratios=True) -> list:
    """
    Extracts lexical features from text and returns them as a vector (list).

    Args:
        text: The input text string.
        include_pos_ratios: Whether to include POS tag ratios in the vector.

    Returns:
        A list containing the calculated feature values in a predefined order,
        or None if the input text is empty/invalid.
    """
    feature_names = list(LEXICAL_FEATURE_NAMES) # Make a copy

    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space and not token.is_punct] # Exclude space/punct for lexical counts
    token_count = float(len(tokens))

    # Denominators for normalization (avoid division by zero)
    safe_token_count = token_count if token_count > 0 else 1.0

    # --- Calculate Features ---
    lower_tokens = [t.text.lower() for t in tokens]
    lemmas = [t.lemma_.lower() for t in tokens]

    ttr = len(set(lower_tokens)) / safe_token_count
    lemma_ttr = len(set(lemmas)) / safe_token_count

    transition_word_count = sum(1 for t in lower_tokens if t in TRANSITION_WORDS)
    hedge_word_count = sum(1 for t in lower_tokens if t in HEDGE_WORDS)

    # Contraction approximation (count based on common suffixes)
    contraction_count = sum(1 for t in lower_tokens if any(t.endswith(suffix) for suffix in CONTRACTIONS_SUFFIXES))
    # A slightly better check might look at the original text:
    # contraction_count = sum(1 for word in text.split() if any(word.lower().endswith(s) for s in CONTRACTIONS_SUFFIXES))


    transition_word_ratio = transition_word_count / safe_token_count
    hedge_word_ratio = hedge_word_count / safe_token_count
    contraction_ratio = contraction_count / safe_token_count # Normalize by token count

    # --- Create Base Feature Vector ---
    feature_vector = [
        ttr,
        lemma_ttr,
        transition_word_ratio,
        hedge_word_ratio,
        contraction_ratio,
    ]

    # --- Optional: Add POS Tag Ratios ---
    if include_pos_ratios:
        # Use all tokens including punctuation for POS counts if desired, or stick to non-punct
        all_tokens_for_pos = [token for token in doc if not token.is_space]
        safe_all_token_count = float(len(all_tokens_for_pos)) if len(all_tokens_for_pos) > 0 else 1.0
        pos_counts = Counter(token.pos_ for token in all_tokens_for_pos)

        # Add POS features in a consistent order (important!)
        # Get all POS tags present in this document
        current_pos_tags = sorted(pos_counts.keys())
        # Append names dynamically if this is the first call or use a predefined list
        if not any(name.startswith('pos_') for name in feature_names):
            for pos in current_pos_tags:
                feature_names.append(f'pos_{pos}_ratio')

        # Append values corresponding to the full feature_names list
        pos_vector_part = []
        present_pos_ratios = {f'pos_{pos}_ratio': count / safe_all_token_count for pos, count in pos_counts.items()}
        for name in feature_names:
            if name.startswith('pos_'):
                pos_vector_part.append(present_pos_ratios.get(name, 0.0)) # Add 0 if tag not in this doc

        feature_vector.extend(pos_vector_part)


    final_feature_vector = [
        ttr,
        lemma_ttr,
        transition_word_ratio,
        hedge_word_ratio,
        contraction_ratio,
    ]


    return final_feature_vector

from typing import Tuple

# Define a standard, fixed list of POS tags (Universal POS Tags)
# You can adjust this list if needed, but keep it consistent.
STANDARD_POS_TAGS = sorted([
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE'
])
# Create the corresponding feature names list once
STANDARD_POS_FEATURE_NAMES = [f'pos_{pos}_ratio' for pos in STANDARD_POS_TAGS]

# If you wanted POS features separately (Recommended pattern):
def extract_pos_features_vector(text: str) -> Tuple[list, list]:
    """
    Extracts POS tag ratios based on a fixed list of standard POS tags.
    """
    if not text or not text.strip():
        # Return empty names (or standard names) and a zero vector of standard length
        return STANDARD_POS_FEATURE_NAMES, [0.0] * len(STANDARD_POS_TAGS)

    doc = nlp(text)
    all_tokens_for_pos = [token for token in doc if not token.is_space]
    safe_all_token_count = float(len(all_tokens_for_pos)) if len(all_tokens_for_pos) > 0 else 1.0

    if safe_all_token_count == 1.0 and len(all_tokens_for_pos) == 0: # Handle truly empty case
         return STANDARD_POS_FEATURE_NAMES, [0.0] * len(STANDARD_POS_TAGS)

    pos_counts = Counter(token.pos_ for token in all_tokens_for_pos)

    # Calculate ratios based on the standard list
    feature_vector = []
    for pos_tag in STANDARD_POS_TAGS:
        count = pos_counts.get(pos_tag, 0) # Get count, default to 0 if tag not present
        feature_vector.append(count / safe_all_token_count)

    # Return the standard names and the fixed-length vector
    return STANDARD_POS_FEATURE_NAMES, feature_vector


import textstat # For readability scores


# Define the order and names of features for the output vector
# We separate dependency ratios for easier handling if needed
CORE_SYNTACTIC_FEATURE_NAMES = [
    "avg_sentence_length_tokens",
    "std_dev_sentence_length_tokens",
    "flesch_reading_ease",
    "gunning_fog",
    "avg_dependency_distance",
    "passive_voice_ratio",
    # Add other core syntactic measures here if desired
]

# Define a list of common dependency relations you might want ratios for
# You can customize this list extensively
COMMON_DEP_RELATIONS = [
    'nsubj', 'dobj', 'iobj', 'csubj', 'ccomp', 'xcomp', # Core arguments
    'obl', 'advcl', 'advmod', # Adverbials / Oblique args
    'amod', 'acl', # Adjectival modifiers
    'det', 'case', 'mark', # Function words related to structure
    'aux', 'auxpass', 'cop', # Auxiliaries / Copula
    'conj', 'cc', # Coordination
    'prep', # Prepositional modifier (often captured within obl etc. depending on scheme)
    'nmod', # Nominal modifier
    'appos', # Appositional modifier
    'ROOT' # Root of the sentence
]

def extract_syntactic_features_vector(text: str, include_dep_ratios=True) -> Tuple[list, list, list]:
    """
    Extracts syntactic features from text and returns them as vectors (list).

    Args:
        text: The input text string.
        include_dep_ratios: Whether to include detailed dependency relation ratios.

    Returns:
        A tuple containing:
          - core_feature_vector: List of core syntactic feature values.
          - dep_ratio_vector: List of dependency ratio values (if requested).
          - dep_ratio_names: List of corresponding dependency ratio names (if requested).
        Returns ([], [], []) if the input text is empty/invalid.
    """
    core_feature_names = list(CORE_SYNTACTIC_FEATURE_NAMES) # Use the predefined core names

    if not text or not text.strip():
        # Return empty lists matching the expected output structure
        dep_names_output = list(f'dep_{dep}_ratio' for dep in COMMON_DEP_RELATIONS) if include_dep_ratios else []
        core_vec_output = [0.0] * len(core_feature_names)
        dep_vec_output = [0.0] * len(dep_names_output) if include_dep_ratios else []
        return core_vec_output, dep_vec_output, dep_names_output
        # return [], [], [] # Simpler empty return

    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    sentences = list(doc.sents)

    token_count = float(len(tokens))
    sentence_count = float(len(sentences))

    # Denominators for normalization
    safe_token_count = token_count if token_count > 0 else 1.0
    safe_sentence_count = sentence_count if sentence_count > 0 else 1.0

    # --- Calculate Core Features ---

    # Sentence Length Stats
    sentence_lengths = [len([t for t in s if not t.is_space]) for s in sentences]
    avg_sentence_length_tokens = np.mean(sentence_lengths) if sentence_lengths else 0.0
    std_dev_sentence_length_tokens = np.std(sentence_lengths) if sentence_lengths else 0.0

    # Readability
    try:
        # Ensure text is not too short for reliable scores
        if len(text.split()) > 50: # Heuristic threshold
             flesch_reading_ease = textstat.flesch_reading_ease(text)
             gunning_fog = textstat.gunning_fog(text)
        else:
             flesch_reading_ease = np.nan # Use NaN for unreliable scores
             gunning_fog = np.nan
    except Exception:
        flesch_reading_ease = np.nan # Use NaN if calculation fails
        gunning_fog = np.nan

    # Dependency Distance
    dep_distances = [abs(t.i - t.head.i) for t in tokens if t.head != t]
    avg_dependency_distance = np.mean(dep_distances) if dep_distances else 0.0

    # Passive Voice Approximation
    passive_count = 0
    verb_tags_count = 0
    for token in tokens:
        # Count verbs (excluding auxiliaries used for passive check later)
        if token.pos_ == "VERB" and token.dep_ != 'aux' and token.dep_ != 'auxpass':
            verb_tags_count += 1
        # Check for passive structure
        if token.dep_ == 'auxpass':
             # Find the verb this auxiliary modifies
             head_verb = token.head
             if head_verb.pos_ == "VERB":
                 # Check if that verb has a passive subject
                 if any(c.dep_ == 'nsubjpass' for c in head_verb.children):
                       passive_count += 1
                 # Handle passive subjects attached higher up (e.g., relative clauses) - more complex
                 # elif head_verb.head.pos_ == "VERB" and any(c.dep_ == 'nsubjpass' for c in head_verb.head.children):
                 #     passive_count += 1


    safe_verb_tags_count = verb_tags_count if verb_tags_count > 0 else 1.0
    passive_voice_ratio = passive_count / safe_verb_tags_count

    # --- Create Core Feature Vector ---
    core_feature_vector = [
        avg_sentence_length_tokens,
        std_dev_sentence_length_tokens,
        flesch_reading_ease,
        gunning_fog,
        avg_dependency_distance,
        passive_voice_ratio,
    ]

    # --- Optional: Calculate Dependency Ratios ---
    dep_ratio_vector = []
    dep_ratio_names = []
    if include_dep_ratios:
        dep_counts = Counter(token.dep_ for token in tokens)
        # Use the predefined list for consistent ordering
        dep_ratio_names = [f'dep_{dep}_ratio' for dep in COMMON_DEP_RELATIONS]
        for dep in COMMON_DEP_RELATIONS:
            dep_ratio_vector.append(dep_counts.get(dep, 0) / safe_token_count)

    # Handle NaN values from readability if necessary (e.g., replace with mean or zero)
    core_feature_vector = [0.0 if np.isnan(x) else x for x in core_feature_vector]


    return core_feature_vector, dep_ratio_vector, dep_ratio_names


# Define lists for specific word categories (reuse from lexical if needed)
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "consequently",
    "nevertheless", "thus", "hence", "otherwise", "indeed", "additionally",
    "subsequently", "finally", "similarly", "likewise", "firstly", "secondly",
    "for example", "in contrast", "on the other hand", "in addition"
    # Add more, including multi-word phrases if desired (requires careful token matching)
}
CONCLUDING_PHRASES = {
    "in conclusion", "to summarize", "in summary", "overall", "to conclude",
    "in short", "finally", "lastly"
}
INTRODUCTORY_PHRASES = {
    "firstly", "to begin with", "initially", "first of all", "the purpose of this"
    # Add more
}


# Define the order and names of features for the output vector
DISCOURSE_FEATURE_NAMES = [
    "transition_word_density_per_sent", # Density of transition words
    "starts_with_intro_phrase",       # Binary: Does text start with intro phrase?
    "ends_with_concluding_phrase",    # Binary: Does last paragraph start with concluding phrase?
    "avg_sentence_position_in_paragraph", # Avg normalized position (0=start, 1=end)
    # Add more sophisticated features below if needed
]

def extract_discourse_structural_features_vector(text: str) -> list:
    """
    Extracts basic discourse and structural features from text
    and returns them as a vector (list).

    Args:
        text: The input text string.

    Returns:
        A list containing the calculated feature values in a predefined order,
        or a list of zeros if the input text is empty/invalid.
    """
    feature_names = list(DISCOURSE_FEATURE_NAMES) # Use the predefined names

    if not text or not text.strip():
        return [0.0] * len(feature_names)

    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    sentences = list(doc.sents)
    # Split paragraphs, keeping only non-empty ones
    paragraphs_text = [p.strip() for p in text.split('\n\n') if p.strip()]

    token_count = float(len(tokens))
    sentence_count = float(len(sentences))
    paragraph_count = float(len(paragraphs_text))

    # Denominators for normalization
    safe_token_count = token_count if token_count > 0 else 1.0
    safe_sentence_count = sentence_count if sentence_count > 0 else 1.0

    # --- Calculate Features ---

    # 1. Transition Word Density (per sentence)
    lower_tokens = [t.text.lower() for t in tokens]
    # Handle multi-word phrases (simple approach: check text contains)
    # A more robust way uses token sequence matching
    transition_word_count = sum(1 for t in lower_tokens if t in TRANSITION_WORDS)
    # Add counts for multi-word phrases (can double count if not careful)
    # for phrase in TRANSITION_WORDS:
    #      if len(phrase.split()) > 1 and phrase in text.lower():
    #          transition_word_count += text.lower().count(phrase) # Simple count

    transition_word_density_per_sent = transition_word_count / safe_sentence_count

    # 2. Starts with Introductory Phrase
    starts_with_intro_phrase = 0.0
    if text.strip():
         # Check if the very beginning matches any phrase
         text_lower_start = text.lower().strip()
         if any(text_lower_start.startswith(p) for p in INTRODUCTORY_PHRASES):
              starts_with_intro_phrase = 1.0

    # 3. Ends with Concluding Phrase
    ends_with_concluding_phrase = 0.0
    if paragraphs_text:
        last_paragraph_lower = paragraphs_text[-1].lower()
        if any(last_paragraph_lower.startswith(p) for p in CONCLUDING_PHRASES):
            ends_with_concluding_phrase = 1.0

    # 4. Average Sentence Position in Paragraph
    total_relative_position = 0.0
    sentences_processed = 0
    for para_text in paragraphs_text:
        para_doc = nlp(para_text)
        para_sentences = list(para_doc.sents)
        num_sents_in_para = len(para_sentences)
        if num_sents_in_para > 1:
            for i, sent in enumerate(para_sentences):
                # Position ranges from 0 (first) to 1 (last)
                relative_position = float(i) / (num_sents_in_para - 1)
                total_relative_position += relative_position
                sentences_processed += 1
        elif num_sents_in_para == 1:
             # Single sentence paragraph, position is ambiguous, maybe 0.5 or skip?
             # Let's assign 0.5 as it's both start and end.
             total_relative_position += 0.5
             sentences_processed += 1

    avg_sentence_position_in_paragraph = (total_relative_position / sentences_processed
                                          if sentences_processed > 0 else 0.0)


    # --- Create Feature Vector ---
    feature_vector = [
        transition_word_density_per_sent,
        starts_with_intro_phrase,
        ends_with_concluding_phrase,
        avg_sentence_position_in_paragraph,
    ]

    return feature_vector

def extract_feature_vector(text):   
    surface_formatting_feature_vec = extract_surface_formatting_features_vector(text)
    lexical_vec = extract_lexical_features_vector(text)
    pos_names, pos_vec = extract_pos_features_vector(text)
    core_vec, dep_vec, dep_names = extract_syntactic_features_vector(text, include_dep_ratios=True)
    discourse_vec = extract_discourse_structural_features_vector(text)
    vec = surface_formatting_feature_vec + lexical_vec + pos_vec + core_vec + dep_vec + discourse_vec
    return vec

def feature_names():
    return SURFACE_FEATURE_NAMES + LEXICAL_FEATURE_NAMES + STANDARD_POS_FEATURE_NAMES + CORE_SYNTACTIC_FEATURE_NAMES + COMMON_DEP_RELATIONS + DISCOURSE_FEATURE_NAMES


def save_feature_vecs(feature_vecs, filename):
    with open(filename+".json", "w") as f:
        json.dump(feature_vecs, f)

    feature_vecs = np.array(feature_vecs, dtype=np.float32)
    np.save(filename+".npy", feature_vecs)


def load_model_and_tokenizer(model_name: str, device: str):
    if "cuda" in device:
        assert torch.cuda.is_available(), "CUDA is not available"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model '{model_name}' loaded on {device}.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        # Handle error appropriately - perhaps skip these features
        tokenizer = None
        model = None
    return tokenizer, model


# Define feature names
MODEL_PROBABILITY_FEATURE_NAMES = [
    "perplexity",
    "average_log_probability",
    "distributional_confidence",
    "self_certainty",
    "per_token_entropy",
    "per_token_dist_perplexity",
    "per_token_gini_impurity",
    "per_token_kl_divergence"
]

def calculate_model_probability_features(text: str, context: str = None, model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda") -> Tuple[float, float, float, float, float, List[float], List[float], List[float], List[float]]:
    """
    Calculate perplexity, average negative log likelihood, average log-probability,
    distributional confidence, self-certainty, and per-token metrics (entropy,
    distributional perplexity, Gini impurity, KL divergence) for a given prompt and answer.
    Perplexity is computed only over the answer tokens, masking out the prompt.
    
    Args:
        text: The target text (e.g., the answer).
        context: Optional preceding text (e.g., the question). If provided,
                 calculates P(text | context). If None, calculates P(text).
        model_name_or_path: The name or path of the pre-loaded transformer model.
        device: The device to run the model on.
        
    Returns:
        Tuple[float, float, float, float, float, List[float], List[float], List[float], List[float]]:
            (perplexity, avg_negative_log_likelihood, avg_log_probability,
             distributional_confidence, self_certainty,
             per_token_entropy, per_token_dist_perplexity, per_token_gini_impurity, per_token_kl_divergence)
    """
    tokenizer, model = load_model_and_tokenizer(model_name_or_path, device)
    try:
        # Set model to evaluation mode
        model.eval()

        question_messages = [
            {"role": "user", "content": context},
        ]
        full_messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": text}
        ]
        question_input_text = tokenizer.apply_chat_template(question_messages, tokenize=False)
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        
        # Tokenize inputs
        encodings = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)
        
        # Get the prompt length to separate prompt and answer tokens
        prompt_encodings = tokenizer(
            question_input_text,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        prompt_length = prompt_encodings["input_ids"].size(1)
        
        # Create labels tensor with prompt tokens masked out
        labels = input_ids.clone()
        # Set labels for prompt tokens to -100 to ignore them in loss computation
        labels[:, :prompt_length] = -100
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Extract logits
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute per-token negative log likelihood
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Reshape loss to match sequence length
            loss = loss.view(shift_labels.size())
            
            # Extract loss for the answer part only (already handled by ignore_index=-100)
            answer_loss = loss[:, prompt_length-1:].mean() if prompt_length < loss.size(1) else loss.mean()
            
            # Calculate perplexity over the answer tokens only
            perplexity = torch.exp(answer_loss) if not torch.isnan(answer_loss) else float('inf')
            
            # Calculate log-probabilities for each token
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Calculate average log-probability for the answer part
            answer_log_probs = token_log_probs[:, prompt_length-1:] if prompt_length < token_log_probs.size(1) else token_log_probs
            avg_log_prob = answer_log_probs.mean()
            
            # Calculate probabilities for additional metrics
            probs = F.softmax(shift_logits, dim=-1)
            
            # Distributional Confidence: Average of max probabilities for the answer part
            max_probs = probs.max(dim=-1).values
            answer_max_probs = max_probs[:, prompt_length-1:] if prompt_length < max_probs.size(1) else max_probs
            distributional_confidence = answer_max_probs.mean()
            
            # Self-Certainty: Variance of max probabilities (lower variance -> higher self-certainty)
            if answer_max_probs.numel() > 1:  # Need at least 2 values to compute variance
                self_certainty = 1.0 / (answer_max_probs.var() + 1e-10)  # Inverse variance as a proxy
            else:
                self_certainty = 1.0  # Default to high certainty if only one token
            
            # Compute per-token metrics for the answer part
            probs_eps = probs + 1e-10  # Add epsilon to avoid log(0)
            
            # Per-token Entropy: -sum(p_i * log(p_i))
            entropy = -(probs_eps * probs_eps.log()).sum(dim=-1)
            answer_entropy = entropy[:, prompt_length-1:] if prompt_length < entropy.size(1) else entropy
            per_token_entropy = answer_entropy.squeeze(0).tolist()
            
            # Per-token Distributional Perplexity: exp(entropy)
            dist_perplexity = torch.exp(answer_entropy)
            per_token_dist_perplexity = dist_perplexity.squeeze(0).tolist()
            
            # Per-token Gini Impurity: 1 - sum(p_i^2)
            squared_probs = probs ** 2
            gini_impurity = 1.0 - squared_probs.sum(dim=-1)
            answer_gini = gini_impurity[:, prompt_length-1:] if prompt_length < gini_impurity.size(1) else gini_impurity
            per_token_gini_impurity = answer_gini.squeeze(0).tolist()
            
            # Per-token KL Divergence w.r.t. uniform distribution
            vocab_size = shift_logits.size(-1)
            uniform_dist = torch.full_like(probs, 1.0 / vocab_size)
            uniform_eps = uniform_dist + 1e-10
            kl_div = (probs_eps * (probs_eps.log() - uniform_eps.log())).sum(dim=-1)
            answer_kl_div = kl_div[:, prompt_length-1:] if prompt_length < kl_div.size(1) else kl_div
            per_token_kl_divergence = answer_kl_div.squeeze(0).tolist()
            
            return (
                perplexity.item(),
                answer_loss.item(),
                avg_log_prob.item(),
                distributional_confidence.item(),
                self_certainty.item(),
                per_token_entropy,
                per_token_dist_perplexity,
                per_token_gini_impurity,
                per_token_kl_divergence
            )
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return float('inf'), float('inf'), float('-inf'), 0.0, 0.0, [], [], [], []


def calculate_metrics_batch(inputs, outputs, model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda"):
    tokenizer, model = load_model_and_tokenizer(model_name_or_path, device)

    results = list()
    for i, o in zip(inputs, outputs):
        single_features = calculate_model_probability_features(i, o, model, tokenizer, max_length=max_length)
        results.append(single_features)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data_path = args.input_file
    output_path = args.output_file

    with open(data_path, "r") as f:
        data = json.load(f)

    feature_vecs = list()
    for item in tqdm(data):
        text = item['output']
        feature_vec = extract_feature_vector(text)
        feature_vecs.append(feature_vec)
    
    save_feature_vecs(feature_vecs, output_path)