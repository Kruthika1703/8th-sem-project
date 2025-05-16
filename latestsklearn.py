import joblib

# Load old models and vectorizers
model = joblib.load("twitter_acc_model.sav")
vectorizer1 = joblib.load("twitter_acc_vectorizer.sav")
classifier = joblib.load("lr_model.sav")
vectorizer = joblib.load("vectorizer.sav")
amaz_model = joblib.load("amazon_acc_model.sav")
amaz_vectorizer = joblib.load("amazon_acc_vectorizer.sav")

# Re-save them using the latest scikit-learn version
joblib.dump(model, "updated_twitter_acc_model.sav")
joblib.dump(vectorizer1, "updated_twitter_acc_vectorizer.sav")
joblib.dump(classifier, "updated_lr_model.sav")
joblib.dump(vectorizer, "updated_vectorizer.sav")
joblib.dump(amaz_model, "updated_amazon_acc_model.sav")
joblib.dump(amaz_vectorizer, "updated_amazon_acc_vectorizer.sav")

print("All models and vectorizers have been updated successfully!")
