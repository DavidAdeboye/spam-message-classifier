import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t')
df.columns = ['label', 'message']

# Preprocess the messages
df['message'] = df['message'].apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and MultinomialNB
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB(alpha=0.1))
])

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train the final model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print accuracy score
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.3f}")

# Save model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel trained and saved successfully as spam_model.pkl")