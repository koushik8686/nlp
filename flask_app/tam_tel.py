import pandas as pd
import json
from collections import Counter
from googletrans import Translator

# Load token mappings and bigram probabilities
token_mappings = pd.read_csv("token_mapping.csv")
tamil_to_telugu = {}
for _, row in token_mappings.iterrows():
    tamil_to_telugu.setdefault(row['Tamil_Token'], []).append(row['Telugu_Token'])

with open("telugu_bigram_probabilities.json", "r", encoding="utf-8") as f:
    telugu_bigram_table = json.load(f)

# Utility functions
def get_bigram_probability(w1, w2, smoothing_factor=0.1):
    """Enhanced bigram probability with smoothing"""
    prob = telugu_bigram_table.get(w1, {}).get(w2, 0)
    return prob if prob > 0 else smoothing_factor

def map_tamil_to_telugu_letters(tamil_word):
    """Enhanced character mapping with compound characters"""
    tamil_to_telugu_mapping = {
        "அ": "అ", "ஆ": "ఆ", "இ": "ఇ", "ஈ": "ఈ", "உ": "ఉ", "ஊ": "ఊ", "எ": "ఎ", "ஏ": "ఏ",
        "ஐ": "ఐ", "ஒ": "ఒ", "ஓ": "ఓ", "ஔ": "ఔ", "க": "క", "ச": "చ", "ஜ": "జ", "ஞ": "ఞ",
        "ட": "ట", "ண": "ణ", "த": "త", "ந": "న", "ப": "ప", "ம": "మ", "ய": "య", "ர": "ర",
        "ல": "ల", "வ": "వ", "ழ": "ళ", "ற": "ర", "ள": "ళ", "ஷ": "ష", "ஸ": "స", "ஹ": "హ",
        "க்": "క్", "ச்": "చ్", "ட்": "ట్", "த்": "త్", "ப்": "ప్", "ற்": "ర్"
    }
    
    result = ""
    i = 0
    while i < len(tamil_word):
        if i < len(tamil_word) - 1 and tamil_word[i:i+2] in tamil_to_telugu_mapping:
            result += tamil_to_telugu_mapping[tamil_word[i:i+2]]
            i += 2
        elif tamil_word[i] in tamil_to_telugu_mapping:
            result += tamil_to_telugu_mapping[tamil_word[i]]
            i += 1
        else:
            result += tamil_word[i]
            i += 1
    return result

def tokenize_tamil_sentence(sentence):
    """Improved tokenization with basic preprocessing"""
    sentence = sentence.strip()
    return [token for token in sentence.split() if token]

def map_tamil_to_telugu_tokens(tamil_tokens):
    """Enhanced token mapping with fallback mechanism"""
    telugu_tokens = []
    for word in tamil_tokens:
        if word in tamil_to_telugu:
            telugu_tokens.append(tamil_to_telugu[word])
        else:
            # Try to find similar tokens if exact match not found
            mapped_word = map_tamil_to_telugu_letters(word)
            similar_tokens = [t for t in telugu_bigram_table.keys() 
                            if calculate_similarity(mapped_word, t) > 0.7]
            if similar_tokens:
                telugu_tokens.append(similar_tokens)
            else:
                telugu_tokens.append([mapped_word])
    return telugu_tokens

def calculate_similarity(word1, word2):
    """Calculate character-level similarity between words"""
    set1 = set(word1)
    set2 = set(word2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def generate_bigrams(telugu_tokens):
    """Enhanced bigram generation with context window"""
    bigrams = []
    context_window = 2  # Consider wider context
    
    for i in range(len(telugu_tokens) - 1):
        for token1 in telugu_tokens[i]:
            # Look ahead within context window
            for j in range(1, min(context_window + 1, len(telugu_tokens) - i)):
                for token2 in telugu_tokens[i + j]:
                    bigrams.append((token1, token2, j))  # Include distance in context
    return bigrams

def calculate_bigram_probabilities(bigrams):
    """Enhanced probability calculation with context weighting"""
    bigram_probs = []
    for w1, w2, distance in bigrams:
        # Apply distance penalty to prefer closer words
        distance_penalty = 1 / (distance + 1)
        prob = get_bigram_probability(w1, w2) * distance_penalty
        bigram_probs.append((w1, w2, prob))
    return bigram_probs

def select_best_bigrams(bigram_probs):
    """Improved bigram selection with backoff strategy"""
    if not bigram_probs:
        return []
        
    sorted_bigrams = sorted(bigram_probs, key=lambda x: x[2], reverse=True)
    translated_tokens = []
    current_token = sorted_bigrams[0][0]
    translated_tokens.append(current_token)
    
    used_pairs = set()
    for i in range(1, len(sorted_bigrams)):
        candidates = [(w1, w2, p) for w1, w2, p in sorted_bigrams 
                     if w1 == current_token and (w1, w2) not in used_pairs]
        
        if candidates:
            best_pair = max(candidates, key=lambda x: x[2])
            best_next_word = best_pair[1]
            used_pairs.add((best_pair[0], best_pair[1]))
            translated_tokens.append(best_next_word)
            current_token = best_next_word
        else:
            # Backoff: choose the most probable next token
            backup_candidates = [(w1, w2, p) for w1, w2, p in sorted_bigrams 
                               if (w1, w2) not in used_pairs]
            if backup_candidates:
                best_backup = max(backup_candidates, key=lambda x: x[2])
                translated_tokens.append(best_backup[1])
                current_token = best_backup[1]
                used_pairs.add((best_backup[0], best_backup[1]))
    
    return translated_tokens

def handle_last_token_and_assemble(tamil_tokens, translated_tokens):
    """Improved token handling with context consideration"""
    if len(tamil_tokens) % 2 != 0:
        last_token = tamil_tokens[-1]
        if last_token in tamil_to_telugu:
            # Choose the most appropriate translation based on previous token
            candidates = tamil_to_telugu[last_token]
            if translated_tokens:
                prev_token = translated_tokens[-1]
                best_prob = -1
                best_candidate = candidates[0]
                
                for candidate in candidates:
                    prob = get_bigram_probability(prev_token, candidate)
                    if prob > best_prob:
                        best_prob = prob
                        best_candidate = candidate
                translated_tokens.append(best_candidate)
            else:
                translated_tokens.append(tamil_to_telugu[last_token][0])
        else:
            mapped_token = map_tamil_to_telugu_letters(last_token)
            translated_tokens.append(mapped_token)
    
    return " ".join(translated_tokens).strip()

def calculate_accuracy(expected_sentence, translated_sentence):
    """Enhanced accuracy calculation with word-level and character-level metrics"""
    # Character-level accuracy
    expected_chars = expected_sentence.replace(" ", "")
    translated_chars = translated_sentence.replace(" ", "")
    
    expected_counter = Counter(expected_chars)
    translated_counter = Counter(translated_chars)
    
    common_chars = sum((expected_counter & translated_counter).values())
    char_accuracy = (common_chars / len(expected_chars)) * 100 if len(expected_chars) > 0 else 0
    
    # Word-level accuracy
    expected_words = expected_sentence.split()
    translated_words = translated_sentence.split()
    
    word_matches = sum(1 for w1, w2 in zip(expected_words, translated_words) if w1 == w2)
    word_accuracy = (word_matches / max(len(expected_words), len(translated_words))) * 100
    
    # Combined accuracy score
    return (char_accuracy * 0.6 + word_accuracy * 0.4)

def get_expected_translation(tamil_sentence):
    """Get expected translation with error handling"""
    try:
        translator = Translator()
        translated = translator.translate(tamil_sentence, src='ta', dest='te')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None