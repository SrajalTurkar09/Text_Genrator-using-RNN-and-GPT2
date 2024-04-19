import streamlit as st
import tensorflow as tf
import numpy as np
import random
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
gpt2_model_name = "gpt2-medium"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = TFGPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Load your existing model
model = tf.keras.models.load_model('poem_model.model')

# Load and preprocess text data
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted((set(text)))
char_to_num = dict((char, i) for i, char in enumerate(characters))
num_to_char = dict((i, char) for i, char in enumerate(characters))
SEQ_LENGTH = 40

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text_with_rnn(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_num[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = num_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def generate_text_with_gpt2(prompt, max_length=50, temperature=0.7):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="tf")
    output = gpt2_model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("Poetry Generation")

method = st.selectbox("Select Model", ("RNN", "GPT-2"))
temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, step=0.1, value=0.7)
length = st.slider("Length", min_value=50, max_value=500, step=50, value=300)

if method == "RNN":
    st.write("Using your existing RNN-based model")
    poem = generate_text_with_rnn(length, temperature)
else:
    st.write("Using pre-trained GPT-2 model")
    poem = generate_text_with_gpt2(random.choice(list(char_to_num.keys())), max_length=length, temperature=temperature)

st.subheader("Generated Poem")
st.write(poem)
