import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
# Load data
df = pd.read_csv("cleaned_file.csv")

# Extract Telugu and Tamil sentences
telugu_sentences = df['Telugu']
tamil_sentences = df['Tamil']

# Function to tokenize sentences
def tokenize_sentences(sentences):
    return [sentence.split() for sentence in sentences]

# Tokenize Telugu and Tamil sentences
tokenized_telugu = tokenize_sentences(telugu_sentences)
tokenized_tamil = tokenize_sentences(tamil_sentences)

# Vocabulary-building function
def build_vocab(tokenized_sentences):
    word_count = Counter([word for sentence in tokenized_sentences for word in sentence])
    return {word: idx for idx, (word, _) in enumerate(word_count.most_common(), start=1)}

# Build vocabularies for both languages
telugu_vocab = build_vocab(tokenized_telugu)
tamil_vocab = build_vocab(tokenized_tamil)
telugu_vocab_df = pd.DataFrame(list(telugu_vocab.items()), columns=['Telugu_Word', 'Telugu_Index'])
tamil_vocab_df = pd.DataFrame(list(tamil_vocab.items()), columns=['Tamil_Word', 'Tamil_Index'])

# Merge the vocabularies into a single DataFrame for easy saving
cdf = pd.concat([telugu_vocab_df, tamil_vocab_df], axis=1)

# Save the combined vocabulary DataFrame to CSV
cdf.to_csv('vocabulary.csv', index=False)
print("Vocabulary has been saved to 'vocabulary.csv'")
# Function to convert sentences to integer sequences
def sentence_to_sequence(sentence, vocab):
    return [vocab.get(word, 0) for word in sentence]  # `0` as a default for unknown words

# Convert tokenized sentences to integer sequences
telugu_sequences = [sentence_to_sequence(sentence, telugu_vocab) for sentence in tokenized_telugu]
tamil_sequences = [sentence_to_sequence(sentence, tamil_vocab) for sentence in tokenized_tamil]

# Define max sequence length and pad sequences
max_seq_length = max(max(len(seq) for seq in telugu_sequences), max(len(seq) for seq in tamil_sequences))
telugu_sequences = pad_sequences(telugu_sequences, maxlen=max_seq_length, padding='post')
tamil_sequences = pad_sequences(tamil_sequences, maxlen=max_seq_length, padding='post')

# Convert sequences to list format to save in a CSV
dtf = pd.DataFrame({
    'tamil_sequences': [list(seq) for seq in tamil_sequences],
    'telugu_sequences': [list(seq) for seq in telugu_sequences]
})

# Save to CSV
dtf.to_csv('sequences.csv', index=False)

print("Sequences have been saved to 'sequences.csv'")


import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input


max_seq_length = max(max(len(seq) for seq in telugu_sequences), max(len(seq) for seq in tamil_sequences))

# Encoder
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(input_dim=len(telugu_vocab) + 1, output_dim=256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(input_dim=len(tamil_vocab) + 1, output_dim=256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(len(tamil_vocab) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare input and target sequences

# Shift target sequences by one for teacher forcing
decoder_input_data = np.array(tamil_sequences)[:, :-1]  # Tamil input shifted left
decoder_output_data = np.expand_dims(np.array(tamil_sequences)[:, 1:], -1)  # Tamil output shifted right

# Train the model
history = model.fit([telugu_sequences, decoder_input_data], decoder_output_data, batch_size=64, epochs=10, validation_split=0.2)
