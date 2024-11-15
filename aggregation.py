import pandas as pd

# Step 1: Read the CSV file
csv_file = "telugu_tamil.csv"
df = pd.read_csv(csv_file)

# Step 2: Read and parse the text file
with open("nlp.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Step 3: Process the text data (assuming two translations per line)
translations = [line.strip().split(",") for line in lines]

for translation in translations:
    temp = translation[0]
    translation[0] = translation[1]
    translation[1] = temp
    
# Step 4: Create a new DataFrame from the parsed translations
new_data = pd.DataFrame(translations, columns=['Telugu', 'Tamil'])

# Step 5: Append the new data to the existing DataFrame
df = pd.concat([new_data , df], ignore_index=True)

# Step 6: Save the updated DataFrame to a new CSV file
df.to_csv("updated_file.csv", index=False)

print("File updated successfully!")
