import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
from collections import Counter
import itertools
# Load dataset
data = pd.read_csv('processed_output.csv')  

# Preprocess text with bigram probabilities and NER tags
def apply_ner_and_tokenize(text):
    text = re.sub(r'\b(సోమరాజు|சிவா)\b', '[PERSON]', text)
    text = re.sub(r'\b(హైదరాబాద్|சென்னை)\b', '[LOCATION]', text)
    return text.split()

# Generate bigrams and compute probabilities
def generate_bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def get_bigram_probabilities(tokenized_texts):
    bigram_counter = Counter(itertools.chain(*[generate_bigrams(text) for text in tokenized_texts]))
    total_bigrams = sum(bigram_counter.values())
    bigram_probs = {bigram: count / total_bigrams for bigram, count in bigram_counter.items()}
    return bigram_probs

# Tokenize with NER and bigram probabilities
tamil_texts = data['Tamil Tokens'].apply(lambda x: apply_ner_and_tokenize(' '.join(eval(x))))
telugu_texts = data['Telugu Tokens'].apply(lambda x: apply_ner_and_tokenize(' '.join(eval(x))))

tamil_bigram_probs = get_bigram_probabilities(tamil_texts)
telugu_bigram_probs = get_bigram_probabilities(telugu_texts)

# Tokenization and padding
tamil_tokenizer = Tokenizer()
telugu_tokenizer = Tokenizer()
tamil_tokenizer.fit_on_texts([' '.join(text) for text in tamil_texts])
telugu_tokenizer.fit_on_texts([' '.join(text) for text in telugu_texts])

tamil_sequences = tamil_tokenizer.texts_to_sequences([' '.join(text) for text in tamil_texts])
telugu_sequences = telugu_tokenizer.texts_to_sequences([' '.join(text) for text in telugu_texts])

max_len_tamil = max(len(seq) for seq in tamil_sequences)
max_len_telugu = max(len(seq) for seq in telugu_sequences)
tamil_sequences = pad_sequences(tamil_sequences, maxlen=max_len_tamil, padding='post')
telugu_sequences = pad_sequences(telugu_sequences, maxlen=max_len_telugu, padding='post')

# Vocabulary sizes
tamil_vocab_size = len(tamil_tokenizer.word_index) + 1
telugu_vocab_size = len(telugu_tokenizer.word_index) + 1

# Transformer model setup
embedding_dim = 512  # Size of embeddings
num_heads = 8       # Number of attention heads
ff_dim = 2048       # Feed-forward network dimension
dropout_rate = 0.1
num_layers = 4      # Number of transformer blocks

# Positional encoding
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Embedding with positional encoding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        x = self.token_embedding(x)
        x += self.pos_encoding[:, :maxlen, :]
        return x

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=dropout_rate):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(d_model),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):  # added default training=False
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)  # Use training argument here
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # Use training argument here
        return self.layernorm2(out1 + ffn_output)

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = PositionalEmbedding(tamil_vocab_size, embedding_dim, max_len_tamil)(encoder_inputs)
encoder_output = encoder_embedding
for _ in range(num_layers):
    encoder_output = TransformerBlock(embedding_dim, num_heads, ff_dim)(encoder_output)

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = PositionalEmbedding(telugu_vocab_size, embedding_dim, max_len_telugu)(decoder_inputs)
decoder_output = decoder_embedding
for _ in range(num_layers):
    decoder_output = TransformerBlock(embedding_dim, num_heads, ff_dim)(decoder_output)

# Dense layer for final predictions
decoder_dense = Dense(telugu_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_output)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

telugu_sequences_target = np.expand_dims(telugu_sequences[:, 1:], -1)
telugu_sequences_input = telugu_sequences[:, :-1]

model.fit(
    [tamil_sequences, telugu_sequences_input],
    telugu_sequences_target,
    batch_size=64,
    epochs=2,
    validation_split=0.2
)

model.save("transformer_tamil_to_telugu_translation_model.h5")
