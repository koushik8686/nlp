# Import necessary libraries
import pandas as pd

# Expanded custom suffix lists for Telugu and Tamil
telugu_suffixes = [
    'లు', 'కి', 'గా', 'ను', 'తో', 'లో', 'కీ', 'యిన', 'నంద',
    'తోడు', 'వారికి', 'మాట్లాడిన', 'చేయు', 'జరగు', 'వాడే',
    'ం', 'ండు', 'ంబు', 'అ', 'అందు', 'అటం', 'అడం',
    'ఇ', 'ఇంచు', 'ఈ', 'ఈడు', 'క', 'కత్తియ', 'కత్తె',
    'కల', 'కాడు', 'కారము', 'కారము', 'కారి', 'కు',
    'గ', 'గల', 'గా', 'గురు', 'జ', 'జీ', 'జీవి',
    'జుడు', 'ట', 'డ', 'డు', 'త', 'తనము', 'త్వము',
    'ద', 'దారుడు', 'ది', 'న', 'న', 'ని', 'ప',
    'పరుడు', 'బ', 'బడి', 'మ', 'మంది', 'మయము', 'మారి',
    'ము', 'ర', 'రాలు', 'రికము', 'ఱ', 'ఱికము', 'ల',
    'ల', 'లాగా', 'లు', 'లేని', 'లో', 'లోపల', 'వ',
    'వంతుడు', 'వరఁకు', 'వరకు', 'వలన', 'వి', 'వి', 'వు',
    'వు'
]

tamil_suffixes = [
    'ம்', 'கள்', 'க்கு', 'இல்', 'தான்', 'உம்', 'வை',
    'யில்', 'வாக்கில்', 'கிறேன்', 'கொள்ளும்', 'இனிக்கும்',
    'அனுபவிக்கும்', 'கிறேன்', 'கொள்கிறேன்', 'கொள்கிறது',
    'கொள்கின்றன', 'கொள்கின்ற', 'என்று', 'என்று',
    'உள்ள', 'உள்ளது', 'உள்ளன', 'என்', 'உயர',
    'ஏற்க', 'ஏற்கனவே', 'என', 'என்ற', 'என்ன',
    'எங்கே', 'என்றால்', 'பாடல்', 'பார்த்து', 'பற்றி',
    'பொன்', 'முதலில்', 'மலர்ந்த', 'வந்த', 'வந்தது',
    'என்று', 'கண்டேன்', 'அவர்கள்', 'ஆகவே', 'சரி',
    'தெளிவாக', 'மனம்', 'நான்', 'நாங்கள்', 'நீ',
    'நீங்கள்', 'பார்க்கும்'
]
telugu_prefixes = [
    'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఎ', 'ఐ', 'ఒ', 'ఓ',
    'స', 'అక్క', 'అయి', 'అప్ప', 'ఎక్క', 'ఉండు', 'కనిపించు'
]

tamil_prefixes = [
    'அ', 'ஆ', 'இ', 'உ', 'எ', 'ஒ', 'ந்', 'மு', 'வெ', 'செ',
    'க', 'த', 'ப', 'நா', 'உள்', 'என', 'உன்', 'தன்'
]
# Function to perform custom lemmatization
def reduce(word, prefixes, suffixes):
    # Check for prefixes
    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix):]  # Remove the prefix

    # Check for suffixes
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]  # Remove the suffix to get the lemma

    return word

def reduce_sentence(tokens, prefixes, suffixes):
    lemmatized_set = set()  # Create an empty set to hold unique lemmatized tokens
    for token in tokens:
        reduced_token = reduce(token, prefixes, suffixes)
        if reduced_token not in lemmatized_set:
            lemmatized_set.add(reduced_token)
    return lemmatized_set  # Return the set of unique lemmatized tokens

# Step 1: Load the tokenized dataset from a text file
with open("tokens.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Ensure there are exactly two lines (one for Tamil, one for Telugu)

# Step 2: Process the lines
tamil_line = lines[0].strip()  # Read Tamil line
telugu_line = lines[1].strip()  # Read Telugu line

# Extract the tokens from each line
tamil_tokens = tamil_line.replace("Tamil Tokens:", "").strip().split(", ")
telugu_tokens = telugu_line.replace("Telugu Tokens:", "").strip().split(", ")

# Debugging: Count tokens before and after reduction
print(f"Original Tamil tokens count: {len(tamil_tokens)}")
print(f"Original Telugu tokens count: {len(telugu_tokens)}")

# Prepare sets to hold lemmatized tokens (unique)
tamil_lemmatized = set(reduce_sentence(tamil_tokens, tamil_prefixes, tamil_suffixes))
telugu_lemmatized = set(reduce_sentence(telugu_tokens, telugu_prefixes, telugu_suffixes))

# Debugging: Count tokens after reduction and set conversion
print(f"Lemmatized Tamil tokens count: {len(tamil_lemmatized)}")
print(f"Lemmatized Telugu tokens count: {len(telugu_lemmatized)}")

# Step 3: Store lemmatized tokens in a text file
with open("lemmatized_tokens.txt", 'w', encoding='utf-8') as file:
    # Write Tamil tokens
    file.write("Tamil Tokens: " + ", ".join(tamil_lemmatized) + "\n")
    # Write Telugu tokens
    file.write("Telugu Tokens: " + ", ".join(telugu_lemmatized) + "\n")

# Output the counts
print("Stemming applied and tokens stored in lemmatized_tokens.txt!")
