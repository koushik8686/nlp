import pandas as pd
import re

# Step 1: Read the CSV file
csv_file = "removed_column.csv"
df = pd.read_csv(csv_file)

# Step 2: Define special characters to remove (without ':' and '-' adjacent to numbers)
special_chars = ['"','–','◆','›',' »' '"""', '•', '.', ',', '!', '@', '#', '$', '%', '^','-',':', '&', '*', '(', ')', '_', '+', '=', '[', ']', '{', '}', ';', '<', '>', '?', '/', '\\', '|', '~', '`']

# Step 3: Function to remove special characters and English letters
def remove_special_chars(text):
    # Remove special characters except ':' and '-' adjacent to numbers
    for char in special_chars:
        text = text.replace(char, '')
    
    # Remove English letters using regex
    text = re.sub(r'[A-Za-z0-9]', '', text)
    # Preserve ':' and '-' if adjacent to numbers
    return text.strip()  # Removing any leading or trailing spaces

# Step 4: Apply the function to the relevant columns
df['Tamil'] = df['Tamil'].apply(lambda x: remove_special_chars(str(x)))
df['Telugu'] = df['Telugu'].apply(lambda x: remove_special_chars(str(x)))

# Step 5: Save the cleaned DataFrame back to CSV
df.to_csv("cleaned_file.csv", index=False)

print("Special characters and English letters removed, file saved as cleaned_file.csv!")
