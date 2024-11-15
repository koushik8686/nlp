# import pandas as pd
# from googletrans import Translator

# # Initialize the Google Translator
# translator = Translator()

# # Load the existing Telugu-to-Tamil mapping CSV
# input_csv = "token_mapping.csv"
# output_csv = "updated_mappings.csv"
# data = pd.read_csv(input_csv)

# # # Ensure the CSV has the correct structure
# # if 'Telugu' not in data.columns or 'Tamil' not in data.columns:
# #     raise ValueError("The input CSV must have 'Telugu' and 'Tamil' columns.")

# # Function to translate Telugu to Tamil
# def translate_to_tamil(word):
#     try:
#         translated = translator.translate(word, src="te", dest="ta")
#         print(translated.text)
#         return translated.text
#     except Exception as e:
#         print(f"Error translating '{word}': {e}")
#         return word  # Fallback to the original word if translation fails

# # Correct the Tamil translations
# data['Tamil_Token'] = data['Telugu_Token'].apply(translate_to_tamil)

# # Save the updated mappings to a new CSV
# data.to_csv(output_csv, index=False, encoding="utf-8")

# print(f"Updated translations have been saved to '{output_csv}'.")


def map_tamil_to_telugu_letters(tamil_word):
    # Define Tamil-to-Telugu letter mapping in a dictionary
    tamil_to_telugu_mapping = {
        # Basic vowels
        "அ": "అ", "ஆ": "ఆ", "இ": "ఇ", "ஈ": "ఈ", "உ": "ఉ", "ஊ": "ఊ", 
        "எ": "ఎ", "ஏ": "ఏ", "ஐ": "ఐ", "ஒ": "ఒ", "ஓ": "ఓ", "ஔ": "ఔ",

        # Consonants
        "க": "క", "ச": "చ", "ஜ": "జ", "ஞ": "ఞ", "ட": "ట", "ண": "ణ", 
        "த": "త", "ந": "న", "ப": "ప", "ம": "మ", "ய": "య", "ர": "ర",
        "ல": "ల", "வ": "వ", "ழ": "ళ", "ற": "ఱ", "ஷ": "ష", "ஸ": "స", 
        "ஹ": "హ",

        # Vowel markers (diacritics)
        "ா": "ా", "ி": "ి", "ீ": "ీ", "ு": "ు", "ூ": "ూ", "ெ": "ె", 
        "ே": "ే", "ை": "ై", "ொ": "ొ", "ோ": "ో", "ௌ": "ౌ", "்": "",

        # Numerals (if needed)
        "௧": "౧", "௨": "౨", "௩": "౩", "௪": "౪", "௫": "౫", 
        "௬": "౬", "௭": "౭", "௮": "౮", "௯": "౯", "௰": "౦",
    }

    # Logic to handle Tamil compound letters
    telugu_word = ""
    for char in tamil_word:
        telugu_char = tamil_to_telugu_mapping.get(char, char)  # Default to original char if no mapping exists
        telugu_word += telugu_char

    return telugu_word

# Example usage:
tamil_sentence = "நீ எங்கு செல்கிறாய்"
telugu_sentence = map_tamil_to_telugu_letters(tamil_sentence)
print("Tamil:", tamil_sentence)
print("Telugu:", telugu_sentence)
