# tam tel

from flask import Flask, render_template, request
from tam_tel import (
    tokenize_tamil_sentence,
    map_tamil_to_telugu_tokens,
    generate_bigrams as tam_tel_bigrams,
    calculate_bigram_probabilities as tam_tel_probabilities,
    select_best_bigrams as tam_tel_select,
    handle_last_token_and_assemble as tam_tel_assemble,
    calculate_accuracy as tam_tel_accuracy,
    get_expected_translation as tam_tel_expected,
)

from tel_tam import (
    tokenize_telugu_sentence,
    map_telugu_to_tamil_tokens,
    generate_bigrams as tel_tam_bigrams,
    calculate_bigram_probabilities as tel_tam_probabilities,
    select_best_bigrams as tel_tam_select,
    handle_last_token_and_assemble as tel_tam_assemble,
    calculate_accuracy as tel_tam_accuracy,
    get_expected_translation as tel_tam_expected,
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    input_sentence = ""
    translated_sentence = ""
    expected_translation = ""
    accuracy = None
    error_message = ""
    translation_direction = "tam_to_tel"  # Default to Tamil-to-Telugu

    if request.method == "POST":
        input_sentence = request.form.get("input_sentence", "").strip()
        translation_direction = request.form.get("translation_direction", "tam_to_tel")

        if not input_sentence:
            error_message = "Please enter a valid sentence."
        else:
            try:
                if translation_direction == "tam_to_tel":
                    # Tamil to Telugu translation process
                    expected_translation = tam_tel_expected(input_sentence)
                    tamil_tokens = tokenize_tamil_sentence(input_sentence)
                    telugu_tokens = map_tamil_to_telugu_tokens(tamil_tokens)
                    bigrams = tam_tel_bigrams(telugu_tokens)
                    bigram_probs = tam_tel_probabilities(bigrams)
                    translated_tokens = tam_tel_select(bigram_probs)
                    translated_sentence = tam_tel_assemble(tamil_tokens, translated_tokens)
                    accuracy = tam_tel_accuracy(expected_translation, translated_sentence)

                elif translation_direction == "tel_to_tam":
                    # Telugu to Tamil translation process
                    expected_translation = tel_tam_expected(input_sentence)
                    telugu_tokens = tokenize_telugu_sentence(input_sentence)
                    tamil_tokens = map_telugu_to_tamil_tokens(telugu_tokens)
                    bigrams = tel_tam_bigrams(tamil_tokens)
                    bigram_probs = tel_tam_probabilities(bigrams)
                    translated_tokens = tel_tam_select(bigram_probs)
                    translated_sentence = tel_tam_assemble(telugu_tokens, translated_tokens)
                    accuracy = tel_tam_accuracy(expected_translation, translated_sentence)

            except Exception as e:
                error_message = f"An error occurred during translation: {e}"

    return render_template(
        "index.html",
        input_sentence=input_sentence,
        translated_sentence=translated_sentence,
        expected_translation=expected_translation,
        accuracy=accuracy,
        error_message=error_message,
        translation_direction=translation_direction,
    )

if __name__ == "__main__":
    app.run(debug=True)
