#tel_tam

import pandas as pd
import json
from collections import Counter
from googletrans import Translator

# Load token mappings and bigram probabilities
token_mappings = pd.read_csv("token_mapping.csv")
telugu_to_tamil = {}
for _, row in token_mappings.iterrows():
    telugu_to_tamil.setdefault(row['Telugu_Token'], []).append(row['Tamil_Token'])

with open("tamil_bigram_probabilities.json", "r", encoding="utf-8") as f:
    tamil_bigram_table = json.load(f)

# Utility functions
def get_bigram_probability(w1, w2, smoothing_factor=0.1):
    """Enhanced bigram probability with smoothing"""
    prob = tamil_bigram_table.get(w1, {}).get(w2, 0)
    return prob if prob > 0 else smoothing_factor

def map_telugu_to_tamil_letters(telugu_word):
    """Character mapping for Telugu to Tamil"""
    telugu_to_tamil_mapping = {
        "అ": "அ", "ఆ": "ஆ", "ఇ": "இ", "ఈ": "ஈ", "ఉ": "உ", "ఊ": "ஊ", "ఎ": "எ", "ఏ": "ஏ",
        "ఐ": "ஐ", "ఒ": "ஒ", "ఓ": "ஓ", "ఔ": "ஔ", "క": "க", "చ": "ச", "జ": "ஜ", "ఞ": "ஞ",
        "ట": "ட", "ణ": "ண", "త": "த", "న": "ந", "ప": "ப", "మ": "ம", "య": "ய", "ర": "ர",
        "ల": "ல", "వ": "வ", "ళ": "ழ", "ష": "ஷ", "స": "ஸ", "హ": "ஹ",
        "క్": "க்", "చ్": "ச்", "త్": "த்", "ప్": "ப்", "ర్": "ற்"
    }
    
    result = ""
    i = 0
    while i < len(telugu_word):
        if telugu_word[i] in telugu_to_tamil_mapping:
            result += telugu_to_tamil_mapping[telugu_word[i]]
            i += 1
        else:
            result += telugu_word[i]
            i += 1
    return result

def tokenize_telugu_sentence(sentence):
    """Tokenize Telugu sentence"""
    sentence = sentence.strip()
    return [token for token in sentence.split() if token]

def map_telugu_to_tamil_tokens(telugu_tokens):
    """Token mapping with fallback for Telugu to Tamil"""
    tamil_tokens = []
    for word in telugu_tokens:
        if word in telugu_to_tamil:
            tamil_tokens.append(telugu_to_tamil[word])
        else:
            mapped_word = map_telugu_to_tamil_letters(word)
            similar_tokens = [t for t in tamil_bigram_table.keys()
                              if calculate_similarity(mapped_word, t) > 0.7]
            if similar_tokens:
                tamil_tokens.append(similar_tokens)
            else:
                tamil_tokens.append([mapped_word])
    return tamil_tokens

def generate_bigrams(tamil_tokens):
    """Generate bigrams with context window"""
    bigrams = []
    context_window = 2
    
    for i in range(len(tamil_tokens) - 1):
        for token1 in tamil_tokens[i]:
            for j in range(1, min(context_window + 1, len(tamil_tokens) - i)):
                for token2 in tamil_tokens[i + j]:
                    bigrams.append((token1, token2, j))
    return bigrams

def calculate_bigram_probabilities(bigrams):
    """Probability calculation with context weighting"""
    bigram_probs = []
    for w1, w2, distance in bigrams:
        distance_penalty = 1 / (distance + 1)
        prob = get_bigram_probability(w1, w2) * distance_penalty
        bigram_probs.append((w1, w2, prob))
    return bigram_probs

def select_best_bigrams(bigram_probs):
    """Bigram selection for optimal translation"""
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
            backup_candidates = [(w1, w2, p) for w1, w2, p in sorted_bigrams 
                               if (w1, w2) not in used_pairs]
            if backup_candidates:
                best_backup = max(backup_candidates, key=lambda x: x[2])
                translated_tokens.append(best_backup[1])
                current_token = best_backup[1]
                used_pairs.add((best_backup[0], best_backup[1]))
    
    return translated_tokens

def handle_last_token_and_assemble(telugu_tokens, translated_tokens):
    """Handle last token and assemble Tamil sentence"""
    if len(telugu_tokens) % 2 != 0:
        last_token = telugu_tokens[-1]
        if last_token in telugu_to_tamil:
            candidates = telugu_to_tamil[last_token]
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
                translated_tokens.append(telugu_to_tamil[last_token][0])
        else:
            mapped_token = map_telugu_to_tamil_letters(last_token)
            translated_tokens.append(mapped_token)
    
    return " ".join(translated_tokens).strip()

def get_expected_translation(telugu_sentence):
    """Translate Telugu sentence to Tamil using Google Translate"""
    try:
        translator = Translator()
        translated = translator.translate(telugu_sentence, src='te', dest='ta')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None
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
