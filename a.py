import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('cleaned_file.csv')

# Add start and end tokens
df['Tamil'] = df['Tamil'].apply(lambda x: f'startseq {x} endseq')
df['Telugu'] = df['Telugu'].apply(lambda x: f'startseq {x} endseq')

# Prepare data
telugu_sentences = df['Telugu'].tolist()
tamil_sentences = df['Tamil'].tolist()

# Tokenize Telugu sentences
telugu_tokenizer = Tokenizer(oov_token='<UNK>')
telugu_tokenizer.fit_on_texts(telugu_sentences)
telugu_sequences = telugu_tokenizer.texts_to_sequences(telugu_sentences)
telugu_vocab_size = len(telugu_tokenizer.word_index) + 1
max_len_telugu = max(len(seq) for seq in telugu_sequences)

# Tokenize Tamil sentences
tamil_tokenizer = Tokenizer(oov_token='<UNK>')
tamil_tokenizer.fit_on_texts(tamil_sentences)
tamil_sequences = tamil_tokenizer.texts_to_sequences(tamil_sentences)
tamil_vocab_size = len(tamil_tokenizer.word_index) + 1
max_len_tamil = max(len(seq) for seq in tamil_sequences)

# Pad sequences
telugu_sequences = pad_sequences(telugu_sequences, maxlen=max_len_telugu, padding='post')
tamil_sequences = pad_sequences(tamil_sequences, maxlen=max_len_tamil, padding='post')

# Prepare decoder data
decoder_input_data = tamil_sequences[:, :-1]
decoder_output_data = tamil_sequences[:, 1:]

# Define model parameters
embedding_dim = 128
lstm_units = 256

# Encoder
encoder_inputs = Input(shape=(max_len_telugu,))
encoder_embedding = Embedding(telugu_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(max_len_tamil - 1,))
decoder_embedding = Embedding(tamil_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(tamil_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    [telugu_sequences, decoder_input_data],
    np.expand_dims(decoder_output_data, -1),
    batch_size=64,
    epochs=1,
    validation_split=0.2
)

# Save the model
model.save("telugu_to_tamil_model.h5")

# Translation function
def translate_sentence(sentence, telugu_tokenizer, tamil_tokenizer, model, max_len_telugu, max_len_tamil):
    # Tokenize and pad the input sentence
    input_seq = telugu_tokenizer.texts_to_sequences([f"startseq {sentence} endseq"])
    input_seq = pad_sequences(input_seq, maxlen=max_len_telugu, padding='post')

    # Get encoder states
    encoder_model = Model(encoder_inputs, [state_h, state_c])
    states_value = encoder_model.predict(input_seq)

    # Initialize target sequence with "startseq"
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tamil_tokenizer.word_index['startseq']

    # Decode the sequence
    decoded_sentence = []
    for _ in range(max_len_tamil - 1):
        output_tokens, h, c = decoder_lstm.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tamil_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'endseq' or not sampled_word:
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Test the translation
test_sentence = "నాకు పొట్ట నొప్పి ఉంది"
translated_sentence = translate_sentence(
    test_sentence, telugu_tokenizer, tamil_tokenizer, model, max_len_telugu, max_len_tamil)
print("Telugu Input:", test_sentence)
print("Tamil Translation:", translated_sentence)
