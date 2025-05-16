import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#  Step 1: Load Dataset
df = pd.read_csv("Twitter_dataset.csv", encoding="ISO-8859-1", header=None)
df.columns = ["sentiment", "id", "date", "query", "username", "text"]
df = df[["sentiment", "text"]]

# Map sentiment values (0 → Negative, 2 → Neutral, 4 → Positive)
sentiment_mapping = {0: -1, 2: 0, 4: 1}
df["sentiment"] = df["sentiment"].map(sentiment_mapping)

#  Step 2: Preprocess Text Data
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\@\w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

#  Step 3: Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["sentiment"]

#  Step 4: Train Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#  Step 5: Evaluate Model
y_pred = classifier.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

#  Step 6: Save Model & Vectorizer as .sav Files
joblib.dump(classifier, "Twitter_model.sav")
joblib.dump(vectorizer, "Twitter_vectorizer.sav")

print("Model and vectorizer saved successfully!")

# # Step 7: Load & Test Model on a Sample
# model = joblib.load("sentiment_model.sav")
# vectorizer = joblib.load("sentiment_vectorizer.sav")

# sample_text = ["I love this product, it's amazing!"]
# cleaned_sample = [clean_text(sample_text[0])]
# sample_vector = vectorizer.transform(cleaned_sample)

# prediction = model.predict(sample_vector)
# print("Predicted Sentiment:", prediction[0])  # -1 (Negative), 0 (Neutral), 1 (Positive)
