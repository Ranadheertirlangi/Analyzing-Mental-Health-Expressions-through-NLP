import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model

import pickle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load your trained GRU model
model = tf.keras.models.load_model("best_modelll.h5", compile=False)

st.title("GRU Text Classification App")

# Text input
user_input = st.text_input("Enter text to classify:")

if st.button("Predict"):
    # Preprocess input here (tokenize, pad, etc.)
    # Example (you must replace this with your own preprocessing):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=231, padding='pre')

    # Predict
    prediction = model.predict(padded)
    prediction_class = 1 if prediction[0][0] >= 0.5 else 0
    st.write("Prediction:", prediction_class)
    st.write("probability : ",prediction)
