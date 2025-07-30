# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1️⃣ Load Dataset
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# 2️⃣ Clean & Prepare
df = df[['Review Text', 'Rating']].dropna()
df = df[df['Review Text'].str.strip() != '']
df = df[df['Rating'] != 3]  # Remove neutral

# Label Sentiment: 1 = Positive, 0 = Negative
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

print(f"Dataset size: {df.shape}")
print(df['Sentiment'].value_counts())

# 3️⃣ Split Data
X = df['Review Text']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save Model & Vectorizer
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to /model/")
