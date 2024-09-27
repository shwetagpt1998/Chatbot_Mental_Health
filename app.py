from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the SVM model
model_path = "svm_model.pkl"  # Ensure this is the correct path to your model
faq_path = "Merged_Conversation.csv"  # Ensure this is the correct path to your FAQ data

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found. Please ensure '{model_path}' exists in the correct path.")

try:
    faq_df = pd.read_csv(faq_path, encoding='ISO-8859-1')
except FileNotFoundError:
    faq_df = pd.DataFrame(columns=['Questions', 'Answers'])
    print(f"Error: FAQ file not found. Please ensure '{faq_path}' exists in the correct path.")

# Knowledge Base for Emotion Responses
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮", "worry": "😟", "love": "❤️", "hate": "😡", "fun": "😄"
}

def build_knowledge_base():
    return {
        'joy': "It's wonderful to hear that you're feeling joyful! 😊",
        'sadness': "I'm really sorry you're feeling this way. 😔",
        'neutral': "It sounds like you’re in a neutral state. 😊",
        'anxiety': "It seems like you might be feeling anxious. 😟",
        'anger': "It sounds like you’re feeling frustrated or angry. 😠",
        'default': "I'm here to help with anything you need. 😊",
        'suicidal': "I'm really sorry you're feeling this way. It's important to talk to someone who can provide the right support.",
        'worry': "It sounds like you might be feeling worried. 😟",
        'love': "It sounds like you’re feeling affectionate or loving. ❤️",
        'hate': "It seems like you might be feeling frustrated or angry. 😡",
        'fun': "It sounds like you’re having fun! 😄"
    }

def handle_greetings(text):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in text.lower() for greeting in greetings):
        return "Hello! How can I assist you today? 😊"
    return None

def handle_faqs(text):
    faq_match = faq_df[faq_df['Questions'].str.contains(text, case=False, na=False)]
    if not faq_match.empty:
        return faq_match['Answers'].values[0]
    return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.getcwd(), 'favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please ensure the model file is available.'}), 500

    data = request.json
    text = data.get('text', '')

    # Handle greetings separately
    greeting_response = handle_greetings(text)
    if greeting_response:
        return jsonify({'emotion': 'neutral', 'response': greeting_response, 'emoji': '😊'})

    # Handle FAQs
    faq_response = handle_faqs(text)
    if faq_response:
        return jsonify({'emotion': 'neutral', 'response': faq_response, 'emoji': '🤔'})

    # Predict emotion and generate response
    try:
        prediction = model.predict([text])[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

    # Handle sensitive topics
    knowledge_base = build_knowledge_base()
    if 'suicidal' in text.lower():
        response = knowledge_base.get('suicidal')
        emoji = '😔'
    else:
        response = knowledge_base.get(prediction, knowledge_base['default'])
        emoji = emotions_emoji_dict.get(prediction, '😊')

    return jsonify({'emotion': prediction, 'response': response, 'emoji': emoji})

if __name__ == '__main__':
    app.run(debug=True)
