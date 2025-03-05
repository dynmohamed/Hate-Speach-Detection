from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import pickle  # Import pickle for loading the tokenizer

app = Flask(__name__)

# Load your model
model = load_model('model.h5')

# Load your tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define your text processing function
def process_text(text):
    stopwords = set(['is', 'a', 'the', 'and', 'or'])  # Add your stopwords
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text_input']
    processed_text = process_text(input_text)
    
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
    
    # Make prediction
    pred = model.predict(padded)
    #pred_label = np.argmax(prediction, axis=1)[0]

    if pred<0.5:  # Adjust this according to your model's output
        result = "Hate and abusive"
    else:
        result = "No hate"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)