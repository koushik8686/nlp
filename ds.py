import pandas as pd
import re
from collections import Counter
import json

# Load and clean dataset
df = pd.read_csv("cleaned_file.csv")

# Function to remove unwanted characters but keep essential ones for Telugu and Tamil
def clean_text(text):
    allowed_chars = r'[\u0C00-\u0C7F\u0B80-\u0BFF]'  # Telugu and Tamil Unicode ranges
    cleaned_text = re.sub(rf'[^{allowed_chars} ]', '', str(text))  # Allow space and language-specific chars
    return cleaned_text.strip()

# Apply cleaning function
df['Tamil'] = df['Tamil'].apply(clean_text)
df['Telugu'] = df['Telugu'].apply(clean_text)

# Function to tokenize text by splitting on spaces (basic tokenization)
def basic_tokenize(text):
    return text.split()  # Simple whitespace-based tokenization

df['Tamil_tokens'] = df['Tamil'].apply(basic_tokenize)
df['Telugu_tokens'] = df['Telugu'].apply(basic_tokenize)

# Improved POS Tagging for Telugu and Tamil
def pos_tagging(tokens, language):
    pos_tags = []
    for token in tokens:
        if language == "Telugu":
            if re.match(r'.*ం$', token):  # Verb ending
                pos_tags.append((token, 'VERB'))
            elif re.match(r'.*లు$', token):  # Plural nouns
                pos_tags.append((token, 'NOUN'))
            elif re.match(r'.*కి$', token):  # Postpositions
                pos_tags.append((token, 'POSTP'))
            elif re.match(r'^[A-Z]', token):  # Proper nouns
                pos_tags.append((token, 'PROPER_NOUN'))
            else:
                pos_tags.append((token, 'NOUN'))
        elif language == "Tamil":
            if re.match(r'.*ல்$', token):  # Locative case ending
                pos_tags.append((token, 'POSTP'))
            elif re.match(r'.*து$', token):  # Nouns (neuter gender)
                pos_tags.append((token, 'NOUN'))
            elif re.match(r'.*கிறார்$', token):  # Verb ending (polite form)
                pos_tags.append((token, 'VERB'))
            elif re.match(r'^[A-Z]', token):  # Proper nouns
                pos_tags.append((token, 'PROPER_NOUN'))
            else:
                pos_tags.append((token, 'NOUN'))
    return pos_tags

# Improved Named Entity Recognition (NER) for Telugu and Tamil
def ner_extraction(tokens):
    named_entities = []
    for token in tokens:
        if token in ['హైదరాబాద్', 'చెన్నై', 'ముంబై', 'Hyderabad', 'Chennai', 'Mumbai']:  # Example locations
            named_entities.append((token, 'LOCATION'))
        elif token in ['గూగుల్', 'మైక్రోసాఫ్ట్', 'ఐఐఐటీ', 'Google', 'Microsoft', 'IIIT']:  # Example organizations
            named_entities.append((token, 'ORGANIZATION'))
        elif token in ['రవి', 'కిరణ్', 'శివ', 'Ravi', 'Kiran', 'Siva']:  # Example names
            named_entities.append((token, 'PERSON'))
        else:
            named_entities.append((token, 'O'))  # 'O' for no specific entity
    return named_entities

# Apply POS tagging and NER to Tamil and Telugu tokens
df['Tamil_POS'] = df['Tamil_tokens'].apply(lambda tokens: pos_tagging(tokens, "Tamil"))
df['Telugu_POS'] = df['Telugu_tokens'].apply(lambda tokens: pos_tagging(tokens, "Telugu"))

df['Tamil_NER'] = df['Tamil_tokens'].apply(ner_extraction)
df['Telugu_NER'] = df['Telugu_tokens'].apply(ner_extraction)

# Save all tokens and their mappings
def save_tokens_mapping(df):
    # Create a list of Telugu and Tamil token pairs
    token_mappings = set()  # Using a set to track unique token pairs
    
    for i in range(len(df)):
        telugu_tokens = df['Telugu_tokens'][i]
        tamil_tokens = df['Tamil_tokens'][i]
        
        # Assuming Telugu and Tamil tokens are aligned by index, create pairs
        for t, tm in zip(telugu_tokens, tamil_tokens):
            token_mappings.add((t, tm))  # Adding token pair as a tuple (no duplicates in a set)
    
    # Convert token mappings into a DataFrame
    mapping_df = pd.DataFrame(list(token_mappings), columns=['Telugu_Token', 'Tamil_Token'])
    
    # Save the token mapping to a CSV file
    mapping_df.to_csv("token_mapping.csv", index=False, encoding="utf-8-sig")
    print("Token mapping saved successfully!")

# Call the function to save 
# Call the function to save token mapping
save_tokens_mapping(df)

# Bigram probabilities
def calculate_bigram_probabilities(tokens):
    unigram_counts = Counter(tokens)
    bigram_counts = Counter(zip(tokens, tokens[1:]))
    bigram_probabilities = {
        (w1, w2): count / unigram_counts[w1] for (w1, w2), count in bigram_counts.items()
    }
    return bigram_probabilities

# Save bigram probabilities in JSON format
def save_bigram_as_json(bigram_probabilities, filename):
    word_bigram_map = {}
    for (w1, w2), prob in bigram_probabilities.items():
        if prob > 0:
            word_bigram_map.setdefault(w1, {})[w2] = prob
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(word_bigram_map, file, ensure_ascii=False, indent=2)

# Calculate bigram probabilities for Tamil and Telugu
all_tamil_tokens = [token for tokens in df['Tamil_tokens'] for token in tokens]
all_telugu_tokens = [token for tokens in df['Telugu_tokens'] for token in tokens]

tamil_bigram_probabilities = calculate_bigram_probabilities(all_tamil_tokens)
telugu_bigram_probabilities = calculate_bigram_probabilities(all_telugu_tokens)

save_bigram_as_json(tamil_bigram_probabilities, "tamil_bigram_probabilities.json")
save_bigram_as_json(telugu_bigram_probabilities, "telugu_bigram_probabilities.json")

# Save POS and NER results
df.to_csv("processed_data_with_pos_ner.csv", index=False, encoding="utf-8-sig")

print("Processing complete! Data, token mappings, and bigram probabilities saved.")
