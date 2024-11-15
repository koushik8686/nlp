import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv('cleaned_file.csv')

# Prepare Telugu and Tamil texts
tamil_sentences = df['Tamil'].tolist()
telugu_sentences = df['Telugu'].tolist()  # Make sure this line is added

# Tokenizers for Telugu and Tamil
telugu_tokenizer = Tokenizer()
tamil_tokenizer = Tokenizer()

# Fit tokenizers on data
telugu_tokenizer.fit_on_texts(telugu_sentences)
tamil_tokenizer.fit_on_texts(tamil_sentences)

# Convert sentences to sequences
telugu_sequences = telugu_tokenizer.texts_to_sequences(telugu_sentences)
tamil_sequences = tamil_tokenizer.texts_to_sequences(tamil_sentences)

# Calculate max length for both sequences
max_len_telugu = max(len(seq) for seq in telugu_sequences)
max_len_tamil = max(len(seq) for seq in tamil_sequences)

# Set max_len to the greater of the two
max_len = max(max_len_telugu, max_len_tamil)

# Padding sequences to the same max length for both languages
telugu_sequences = pad_sequences(telugu_sequences, maxlen=max_len, padding='post')
tamil_sequences = pad_sequences(tamil_sequences, maxlen=max_len, padding='post')

# Optional: Print shapes to verify the padding
print("Telugu Sequences Shape:", telugu_sequences.shape)
print("Tamil Sequences Shape:", tamil_sequences.shape)

# Hyperparameters
embedding_dim = 256
lstm_units = 512

# Encoder
encoder_inputs = Input(shape=(max_len,), name="Encoder_Inputs")
encoder_embedding = Embedding(input_dim=len(telugu_tokenizer.word_index) + 1, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True, name="Encoder_LSTM")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(max_len,), name="Decoder_Inputs")
decoder_embedding = Embedding(input_dim=len(tamil_tokenizer.word_index) + 1, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name="Decoder_LSTM")
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(len(tamil_tokenizer.word_index) + 1, activation='softmax', name="Dense_Output")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create decoder inputs and outputs
decoder_input_data = np.zeros_like(tamil_sequences)
decoder_output_data = np.zeros_like(tamil_sequences)

# Shift `tamil_sequences` by one for decoder input and output
decoder_input_data[:, 1:] = tamil_sequences[:, :-1]  # Shift right for decoder input
decoder_output_data[:, :-1] = tamil_sequences[:, 1:]  # Shift left for decoder output

# Print the shapes to check
print("Decoder Input Shape:", decoder_input_data.shape)
print("Decoder Output Shape:", decoder_output_data.shape)

# Now, the model should expect (None, 91) for both decoder_input_data and decoder_output_data.

from tensorflow.keras import backend as K
K.clear_session()

# Re-initialize the model here
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [telugu_sequences, decoder_input_data],
    decoder_output_data,
    batch_size=32,
    epochs=20,
    validation_split=0.2
)
def translate_sentence(sentence):
    # Tokenize and pad the input sentence
    input_seq = telugu_tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_len_telugu, padding='post')

    # Encode the input
    states_value = encoder_lstm.predict(input_seq)

    # Start decoding with the `<start>` token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tamil_tokenizer.word_index['<start>']

    # Generate the sequence
    decoded_sentence = ''
    for _ in range(max_len_tamil):
        output_tokens, h, c = decoder_lstm.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tamil_tokenizer.index_word[sampled_token_index]
        if sampled_word == '<end>':
            break

        decoded_sentence += ' ' + sampled_word

        # Update the target sequence and states
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence

# After training the model
model.save('seq2seq_model.h5')
