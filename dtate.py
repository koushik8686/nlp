import pandas as pd
import json
from collections import Counter

# Load the Tamil-to-Telugu mappings
token_mappings = pd.read_csv("token_mapping.csv")
tamil_to_telugu = {}
for _, row in token_mappings.iterrows():
    tamil_to_telugu.setdefault(row['Tamil_Token'], []).append(row['Telugu_Token'])

# Load Telugu bigram probabilities as a nested dictionary
with open("telugu_bigram_probabilities.json", "r", encoding="utf-8") as f:
    telugu_bigram_table = json.load(f)

# Function to get bigram probability from the JSON table
def get_bigram_probability(w1, w2):
    return telugu_bigram_table.get(w1, {}).get(w2, 0)

# Function to map Tamil letters to Telugu letters when no direct token mapping is found
def map_tamil_to_telugu_letters(tamil_word):
    tamil_to_telugu_mapping = {
        "அ": "అ", "ஆ": "ఆ", "இ": "ఇ", "ஈ": "ఈ", "உ": "ఉ", "ஊ": "ఊ", "எ": "ఎ", "ஏ": "ఈ",
        "ஐ": "ఐ", "ஒ": "ఒ", "ஓ": "ఓ", "ஔ": "ఔ", "க": "క", "ச": "చ", "ஜ": "జ", "ஞ": "ఞ",
        "ட": "ట", "ண": "ణ", "த": "త", "ந": "న", "ப": "ప", "ம": "మ", "ய": "య", "ர": "ర",
        "ல": "ల", "வ": "వ", "ழ": "ళ", "ற": "ర", "ல": "ల", "ஷ": "ష", "ஸ": "స", "ஹ": "హ"
    }
    
    return ''.join([tamil_to_telugu_mapping.get(char, char) for char in tamil_word])

# Step 1: Tokenize the Tamil sentence
def tokenize_tamil_sentence(sentence):
    return sentence.split()  # Basic split by space (consider using a more advanced tokenizer)

# Step 2: Map Tamil words to Telugu tokens
def map_tamil_to_telugu_tokens(tamil_tokens):
    telugu_tokens = []
    for word in tamil_tokens:
        if word in tamil_to_telugu:
            telugu_tokens.append(tamil_to_telugu[word])
        else:
            telugu_tokens.append([map_tamil_to_telugu_letters(word)])
    return telugu_tokens

# Step 3: Generate bigrams from the Telugu tokens
def generate_bigrams(telugu_tokens):
    bigrams = []
    for i in range(len(telugu_tokens) - 1):
        for token1 in telugu_tokens[i]:
            for token2 in telugu_tokens[i + 1]:
                bigrams.append((token1, token2))
    return bigrams

# Step 4: Calculate bigram probabilities
def calculate_bigram_probabilities(bigrams):
    bigram_probs = []
    for w1, w2 in bigrams:
        prob = get_bigram_probability(w1, w2)
        bigram_probs.append((w1, w2, prob))
    return bigram_probs

# Step 5: Choose the best bigram with the highest probability for each position
def select_best_bigrams(bigram_probs):
    sorted_bigrams = sorted(bigram_probs, key=lambda x: x[2], reverse=True)
    translated_tokens = []
    current_token = None
    
    if sorted_bigrams:
        translated_tokens.append(sorted_bigrams[0][0])  # First token from the highest probability bigram
        current_token = sorted_bigrams[0][0]
    
    for i in range(1, len(sorted_bigrams)):
        best_next_word = None
        max_prob = 0
        for w1, w2, prob in sorted_bigrams:
            if w1 == current_token and prob > max_prob:
                best_next_word = w2
                max_prob = prob
        
        if best_next_word:
            translated_tokens.append(best_next_word)
            current_token = best_next_word  # Update the current token
    
    return translated_tokens

# Step 6: Handle last token for odd number of tokens and final translation assembly
def handle_last_token_and_assemble(tamil_tokens, translated_tokens):
    if len(tamil_tokens) % 2 != 0:
        last_token = tamil_tokens[-1]
        if last_token in tamil_to_telugu:
            translated_tokens.append(tamil_to_telugu[last_token][0])
        else:
            translated_tokens.append(map_tamil_to_telugu_letters(last_token)[0])
    
    return " ".join(translated_tokens).strip()

# Function to calculate character-wise accuracy
def calculate_accuracy(tamil_sentence, translated_sentence):
    # Remove spaces and compare character-by-character
    tamil_sentence = tamil_sentence.replace(" ", "")
    translated_sentence = translated_sentence.replace(" ", "")
    
    # Count characters in both sentences
    tamil_counter = Counter(tamil_sentence)
    translated_counter = Counter(translated_sentence)
    
    # Calculate number of matching characters
    common_chars = sum((tamil_counter & translated_counter).values())  # Intersection count
    
    # Calculate accuracy as percentage
    accuracy = (common_chars / len(tamil_sentence)) * 100 if len(tamil_sentence) > 0 else 0
    return accuracy

# Main function to translate a sentence
def translate_sentence(tamil_sentence, expected_telugu_sentence):
    tamil_tokens = tokenize_tamil_sentence(tamil_sentence)
    telugu_tokens = map_tamil_to_telugu_tokens(tamil_tokens)
    
    bigrams = generate_bigrams(telugu_tokens)
    bigram_probs = calculate_bigram_probabilities(bigrams)
    
    translated_tokens = select_best_bigrams(bigram_probs)
    
    translated_sentence = handle_last_token_and_assemble(tamil_tokens, translated_tokens)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(expected_telugu_sentence, translated_sentence)
    
    return translated_sentence, accuracy

# Example usage
tamil_sentence = "நான் உன்னை இன்று பார்க்க விரும்புகிறேன்"
expected_telugu_sentence = "ఈ పుస్తకం బాగా ఉంది"
translated_sentence, accuracy = translate_sentence(tamil_sentence, expected_telugu_sentence)

print("Tamil:", tamil_sentence)
print("Expected Telugu:", expected_telugu_sentence)
print("Translated Telugu:", translated_sentence)
print("Character-wise Accuracy:", accuracy, "%")
