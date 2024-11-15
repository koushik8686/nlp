import pandas as pd
import re

# Load the dataset
df = pd.read_csv("cleaned_file.csv")

# Remove duplicates to ensure each Tamil entry has a unique Telugu translation
df = df.drop_duplicates(subset=['Tamil', 'Telugu']).reset_index(drop=True)

# Define a function for tokenization

def word_tokenize(text):
    # Updated regex to capture words in both Tamil and Telugu scripts
    return list(re.findall(r'[\u0B80-\u0BFF\u0C00-\u0C7F]+', str(text)))

text_tamil = "நాకు பொட்டு நொப்பி உள்ளது"
text_telugu = "నాకు పొట్ట నొప్పి ఉంది"

# Tokenizing Tamil and Telugu texts
tamil_tokens = word_tokenize(text_tamil)
telugu_tokens = word_tokenize(text_telugu)

print("Tamil Tokens:", tamil_tokens)
print("Telugu Tokens:", telugu_tokens)


# Tokenize Tamil and Telugu columns while keeping alignment
df['Tamil_tokens'] = df['Tamil'].apply(word_tokenize)
df['Telugu_tokens'] = df['Telugu'].apply(word_tokenize)

# Re-verify alignment
aligned_df = pd.DataFrame({
    "Tamil Tokens": [' '.join(tokens) for tokens in df['Tamil_tokens']],
    "Telugu Tokens": [' '.join(tokens) for tokens in df['Telugu_tokens']]
})

# Save aligned tokens to a CSV file
aligned_df.to_csv("aligned_tokens.csv", index=False, encoding="utf-8")

print("Tokenization completed with duplicates removed, and tokens stored in aligned_tokens.csv.")
