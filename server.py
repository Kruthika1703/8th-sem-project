from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from googletrans import Translator
import joblib
import asyncio  # Import asyncio

app = Flask(__name__)
CORS(app, resources={r"/predictTweet": {"origins": "http://localhost:49443"}})
CORS(app, resources={r"/predictYT": {"origins": "http://localhost:49443"}})
CORS(app, resources={r"/predictAmazon": {"origins": "http://localhost:49443"}})

model = joblib.load('Twitter_model.sav')
vectorizer1 = joblib.load('Twitter_vectorizer.sav')
classifier = joblib.load('updated_lr_model.sav')
vectorizer = joblib.load('updated_vectorizer.sav')
amaz_model=joblib.load('updated_amazon_acc_model.sav')
amaz_vectorizer=joblib.load('updated_amazon_acc_vectorizer.sav')





sentiments = SentimentIntensityAnalyzer()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
translator = Translator()

stop_words = stopwords.words('english')

def clean_review(review):
    review = " ".join(word for word in review.split() if word not in stop_words)
    return review

def preprocess_text(text):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text






@app.route('/predictTweet', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        user_input = data['text']

        # Translate text to English (Synchronous Call, No asyncio)
        translated_text = translator.translate(user_input, src='auto', dest='en').text

        print(f"Translated Text: {translated_text}")  # Debugging

        # Preprocessing
        cleaned_tweet = translated_text.lower().strip()

        print(f"Cleaned Tweet: {cleaned_tweet}")  # Debugging

        # Calculate VADER sentiment scores
        vader_scores = sentiments.polarity_scores(cleaned_tweet)
        compound_score = vader_scores['compound']

        # Convert to vector and predict sentiment
        tweet_vector = vectorizer1.transform([cleaned_tweet]).toarray()
        sentiment = model.predict(tweet_vector)[0]

        print(f"Model Prediction: {sentiment}, Compound Score: {compound_score}")  # Debugging

        # Adjust sentiment based on VADER score
        if -0.2 <= compound_score <= 0.2:  
            sentiment_label = "Neutral"
        else:
            sentiment_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
            sentiment_label = sentiment_mapping.get(sentiment, "Unknown")

        return jsonify({'sentiment': sentiment_label, 'vader_score': vader_scores})

    except Exception as e:
        return jsonify({'error': str(e)})




@app.route('/predictYT', methods=["POST"])
def predict_sentimentYT():
    try:
        data = request.get_json()
        comment = data['comment']

        # Translate synchronously (No asyncio)
        translated = translator.translate(comment, src='auto', dest='en')
        comment = translated.text  # Extract translated text

        # Preprocessing
        cleaned_comment = ' '.join([w for w in comment.split() if len(w) > 3])
        cleaned_comment = cleaned_comment.lower()

        # Vectorization and Prediction
        comment_vector = vectorizer.transform([cleaned_comment]).toarray()
        sentiment = classifier.predict(comment_vector)
        
        sentiment_maps = {0: "Neutral", -1: "Negative", 1: "Positive"}
        label = sentiment_maps.get(sentiment[0], "Unknown")

        # VADER Sentiment Scores
        vader_scores = sentiments.polarity_scores(cleaned_comment)

        return jsonify({'sentiment': label, 'vader_score': vader_scores})
    except Exception as e:
        return jsonify({'error': str(e)})







@app.route('/predictAmazon', methods=["POST"])
def predict_sentimentAmazon():
    try:
        data = request.get_json()
        review = data['review']

        # Translate text to English (No need for asyncio)
        translated_text = translator.translate(review, src='auto', dest='en').text

        cleaned_review = ' '.join([w for w in translated_text.split() if len(w) > 3])
        cleaned_review = cleaned_review.lower()

        review_vector = amaz_vectorizer.transform([cleaned_review]).toarray()
        sentiment = amaz_model.predict(review_vector)

        sentiment_labels = {0: "Neutral", -1: "Negative", 1: "Positive"}
        label = sentiment_labels.get(sentiment[0], "Unknown")

        vader_scores = sentiments.polarity_scores(cleaned_review)

        return jsonify({'sentiment': label, 'vader_score': vader_scores})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)