import pandas as pd
import json

# Load the Telugu-to-Tamil mappings
token_mappings = pd.read_csv("token_mapping.csv")
telugu_to_tamil = dict(zip(token_mappings['Telugu_Token'], token_mappings['Tamil_Token']))

# Load Tamil bigram probabilities as a nested dictionary
with open("tamil_bigram_probabilities.json", "r", encoding="utf-8") as f:
    tamil_bigram_table = json.load(f)

# Function to get bigram probability from the JSON table
def get_bigram_probability(w1, w2):
    return tamil_bigram_table.get(w1, {}).get(w2, 0)  # Default to 0 if not found

# Function to translate a sentence
def translate_sentence(sentence):
    # Step 1: Tokenize the Telugu sentence
    telugu_tokens = sentence.split()
    
    # Step 2: Map each Telugu word to Tamil, defaulting to the Telugu word if no mapping is found
    tamil_tokens = [telugu_to_tamil.get(word, word) for word in telugu_tokens]
    
    # Debugging output: print the Telugu and Tamil tokens
    print(f"Telugu tokens: {telugu_tokens}")
    print(f"Tamil tokens (after mapping): {' '.join(tamil_tokens)}")
    
    # Step 3: Adjust using bigram probabilities without backtracking
    used_tokens = set()  # Track used Tamil tokens to avoid repetition
    translated_tokens = []  # To store the final sequence of Tamil tokens
    
    for i in range(len(tamil_tokens)):
        current_tamil_word = tamil_tokens[i]
        best_next_word = None
        max_prob = 0
        
        # Find the word with the highest bigram probability that follows the current word
        for next_word in tamil_bigram_table.get(current_tamil_word, {}).keys():
            if next_word not in used_tokens:  # Avoid repetition of tokens
                bigram_prob = get_bigram_probability(current_tamil_word, next_word)
                if bigram_prob > max_prob:
                    max_prob = bigram_prob
                    best_next_word = next_word
        
        if best_next_word is not None:
            translated_tokens.append(best_next_word)
            used_tokens.add(best_next_word)
        else:
            translated_tokens.append(current_tamil_word)  # If no valid next word, keep the current one
    
    # Step 4: Combine tokens into a Tamil sentence
    translated_sentence = " ".join(translated_tokens).strip()
    return translated_sentence

# Example usage
telugu_sentence = "నాకు పొట్ట నొప్పి ఉంది"
translated_sentence = translate_sentence(telugu_sentence)
print("Telugu:", telugu_sentence)
print("Tamil:", translated_sentence)
