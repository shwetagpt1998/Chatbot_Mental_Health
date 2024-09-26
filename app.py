from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

# Initialize Flask App
app = Flask(__name__)

# Load the SVM model - make sure the model is trained and saved with the correct scikit-learn version
try:
    model = joblib.load("svm_model.pkl")  # Ensure this is the correct path to your model
except FileNotFoundError:
    model = None
    print("Error: Model file not found. Please ensure 'svm_model.pkl' exists in the correct path.")

# Load datasets with specified encoding to handle Unicode errors
try:
    faq_df = pd.read_csv("Merged_Conversation.csv", encoding='ISO-8859-1')  # Use the encoding that works for your data
except FileNotFoundError:
    faq_df = pd.DataFrame(columns=['Questions', 'Answers'])  # Fallback to empty DataFrame if file not found
    print("Error: FAQ file not found. Please ensure 'Merged_Conversation.csv' exists in the correct path.")

# Knowledge Base for Emotion Responses
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮", "worry": "😟", "love": "❤️", "hate": "😡", "fun": "😄"
}

def build_knowledge_base():
    return {
        'joy': "It's wonderful to hear that you're feeling joyful! 😊 Sometimes sharing your happiness can make it even more special. If you want to talk more about what’s making you happy, I’m here to listen!",
        'sadness': "I'm really sorry you're feeling this way. 😔 It's important to acknowledge your feelings. Talking about what’s bothering you can help, and I'm here to support you through this.",
        'neutral': "It sounds like you’re in a neutral state. Sometimes, just having a chat can be a good way to change things up. If there's anything specific on your mind, I'm all ears!",
        'anxiety': "It seems like you might be feeling anxious. 😟 It's okay to feel this way, and talking about what's making you anxious might help. I’m here to support you.",
        'anger': "It sounds like you’re feeling frustrated or angry. 😠 Expressing your feelings can be a great way to process them. If you want to talk about what’s causing this anger, I’m here to help.",
        'default': "I'm here to help with anything you need. 😊 If you have any questions or just want to talk, feel free to share!",
        'suicidal': "I'm really sorry you're feeling this way. It's important to talk to someone who can provide the right support, such as a mental health professional. If you need immediate help, please contact a crisis hotline or go to the nearest emergency room.",
        'worry': "It sounds like you might be feeling worried. 😟 Sometimes sharing your concerns can be a good way to lighten the load. If you want to talk about what's worrying you, I’m here to listen.",
        'love': "It sounds like you’re feeling affectionate or loving. ❤️ Expressing love and kindness is wonderful. If there’s something you’d like to share or discuss, I’m here to listen.",
        'hate': "It seems like you might be feeling frustrated or angry. 😡 Talking about what’s causing this hatred might help. I’m here to support you.",
        'fun': "It sounds like you’re having fun! 😄 It's great to hear you’re enjoying yourself. If you want to share more about what’s making you happy, I’m here to chat!"
    }

def handle_greetings(text):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in text.lower() for greeting in greetings):
        return "Hello! How can I assist you today? 😊"
    return None

def handle_faqs(text):
    # Match FAQ questions in the merged dataset
    faq_match = faq_df[faq_df['Questions'].str.contains(text, case=False, na=False)]
    if not faq_match.empty:
        return faq_match['Answers'].values[0]
    return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

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
    prediction = model.predict([text])[0]

    # Handle sensitive topics
    knowledge_base = build_knowledge_base()
    if 'suicidal' in text.lower():
        response = knowledge_base.get('suicidal')
        emoji = '😔'  # Use a neutral or empathetic emoji for sensitive topics
    else:
        response = knowledge_base.get(prediction, knowledge_base['default'])
        emoji = emotions_emoji_dict.get(prediction, "😊")  # Default to smiling emoji if no match

    return jsonify({'emotion': prediction, 'response': response, 'emoji': emoji})

if __name__ == '__main__':
    app.run(debug=True)
