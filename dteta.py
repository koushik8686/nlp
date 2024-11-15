import pandas as pd
import json
from collections import Counter

# Load the Telugu-to-Tamil mappings
token_mappings = pd.read_csv("token_mapping.csv")
telugu_to_tamil = {}
for _, row in token_mappings.iterrows():
    telugu_to_tamil.setdefault(row['Telugu_Token'], []).append(row['Tamil_Token'])

# Load Tamil bigram probabilities as a nested dictionary
with open("tamil_bigram_probabilities.json", "r", encoding="utf-8") as f:
    tamil_bigram_table = json.load(f)

# Function to get bigram probability from the JSON table
def get_tamil_bigram_probability(w1, w2):
    return tamil_bigram_table.get(w1, {}).get(w2, 0)

# Function to map Telugu letters to Tamil letters when no direct token mapping is found
def map_telugu_to_tamil_letters(telugu_word):
    telugu_to_tamil_mapping = {
        "అ": "அ", "ఆ": "ஆ", "ఇ": "இ", "ఈ": "ஈ", "ఉ": "உ", "ఊ": "ஊ", "ఎ": "எ", "ఏ": "ஏ",
        "ఐ": "ஐ", "ఒ": "ஒ", "ఓ": "ஓ", "ఔ": "ஔ", "క": "க", "చ": "ச", "జ": "ஜ", "ఞ": "ஞ",
        "ట": "ட", "ణ": "ண", "త": "த", "న": "ந", "ప": "ப", "మ": "ம", "య": "ய", "ర": "ர",
        "ల": "ல", "వ": "வ", "ళ": "ழ", "ష": "ஷ", "స": "ஸ", "హ": "ஹ"
    }
    
    return ''.join([telugu_to_tamil_mapping.get(char, char) for char in telugu_word])

# Step 1: Tokenize the Telugu sentence
def tokenize_telugu_sentence(sentence):
    return sentence.split()  # Basic split by space

# Step 2: Map Telugu words to Tamil tokens
def map_telugu_to_tamil_tokens(telugu_tokens):
    tamil_tokens = []
    for word in telugu_tokens:
        if word in telugu_to_tamil:
            tamil_tokens.append(telugu_to_tamil[word])
        else:
            tamil_tokens.append([map_telugu_to_tamil_letters(word)])
    return tamil_tokens

# Step 3: Generate bigrams from the Tamil tokens
def generate_bigrams(tamil_tokens):
    bigrams = []
    for i in range(len(tamil_tokens) - 1):
        for token1 in tamil_tokens[i]:
            for token2 in tamil_tokens[i + 1]:
                bigrams.append((token1, token2))
    return bigrams

# Step 4: Calculate bigram probabilities
def calculate_bigram_probabilities(bigrams):
    bigram_probs = []
    for w1, w2 in bigrams:
        prob = get_tamil_bigram_probability(w1, w2)
        bigram_probs.append((w1, w2, prob))
    
    return bigram_probs

# Step 5: Choose the best bigram with the highest probability for each position
def select_best_bigrams(bigram_probs):
    sorted_bigrams = sorted(bigram_probs, key=lambda x: x[2], reverse=True)
    translated_tokens = []
    current_token = None
    print(sorted_bigrams)
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
def handle_last_token_and_assemble(telugu_tokens, translated_tokens):
    if len(telugu_tokens) % 2 != 0:
        last_token = telugu_tokens[-1]
        if last_token in telugu_to_tamil:
            translated_tokens.append(telugu_to_tamil[last_token][0])
        else:
            translated_tokens.append(map_telugu_to_tamil_letters(last_token)[0])
    
    return " ".join(translated_tokens).strip()

# Function to calculate character-wise accuracy
def calculate_accuracy(telugu_sentence, translated_sentence):
    # Remove spaces and compare character-by-character
    telugu_sentence = telugu_sentence.replace(" ", "")
    translated_sentence = translated_sentence.replace(" ", "")
    
    # Count characters in both sentences
    telugu_counter = Counter(telugu_sentence)
    translated_counter = Counter(translated_sentence)
    
    # Calculate number of matching characters
    common_chars = sum((telugu_counter & translated_counter).values())  # Intersection count
    
    # Calculate accuracy as percentage
    accuracy = (common_chars / len(telugu_sentence)) * 100 if len(telugu_sentence) > 0 else 0
    return accuracy

# Main function to translate a sentence
def translate_telugu_to_tamil(telugu_sentence, expected_tamil_sentence):
    telugu_tokens = tokenize_telugu_sentence(telugu_sentence)
    tamil_tokens = map_telugu_to_tamil_tokens(telugu_tokens)
    
    bigrams = generate_bigrams(tamil_tokens)
    bigram_probs = calculate_bigram_probabilities(bigrams)
    
    translated_tokens = select_best_bigrams(bigram_probs)
    
    translated_sentence = handle_last_token_and_assemble(telugu_tokens, translated_tokens)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(expected_tamil_sentence, translated_sentence)
    
    return translated_sentence, accuracy

# Example usage
telugu_sentence = "నాకు పొట్ట నొప్పి ఉంది"
expected_tamil_sentence = "நான் உன்னை பார்க்க விரும்புகிறேன்"
translated_sentence, accuracy = translate_telugu_to_tamil(telugu_sentence, expected_tamil_sentence)

print("Telugu:", telugu_sentence)
print("Expected Tamil:", expected_tamil_sentence)
print("Translated Tamil:", translated_sentence)
print("Character-wise Accuracy:", accuracy, "%")
