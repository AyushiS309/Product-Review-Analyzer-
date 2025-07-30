# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
import numpy as np

# Load model & vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load dataset for EDA
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df[['Review Text', 'Rating']].dropna()
df = df[df['Review Text'].str.strip() != '']
df = df[df['Rating'] != 3]
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# ✅ Lavender Pink background + Persian Pink text theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FBAED2; /* Lavender Pink background */
        color: #D6336C; /* Persian Pink text */
    }

    h1, h2, h3, h4, h5, h6,
    p, label, div, span, .stMarkdown, .css-10trblm {
        color: #D6336C !important;
    }

    .stButton>button {
    background-color: #FBAED2; /* Persian Pink background */
    color: #FBAED2; /* Lavender Pink text */
    border-radius: 12px;
    padding: 0.5em 1.2em;
    border: none;
    font-weight: bold;
    }

    /* Sidebar text */
    .css-1d391kg, .css-1v0mbdj {
        color: #D6336C !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ App title
st.title("🌸 Product Review Analyzer")

# Sidebar navigation
tab = st.sidebar.radio(
    "Choose an option",
    ["🔍 Analyze a Review", "📊 EDA & Visuals"]
)

# 🔍 1️⃣ Analyze a Review
if tab == "🔍 Analyze a Review":
    st.header("✨ Analyze Your Product Review")
    user_input = st.text_area("Write your product review here:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please write something!")
        else:
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)
            prob = np.max(model.predict_proba(transformed))

            if prediction[0] == 1:
                st.success(f"💖 Positive Review ({prob:.2%} confidence)")
            else:
                st.error(f"💔 Negative Review ({prob:.2%} confidence)")

# 📊 2️⃣ EDA & Visuals
else:
    st.header("📊 Data Insights & Visuals")

    st.subheader("⭐ Rating Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Rating', data=df, palette='pastel', ax=ax1)
    st.pyplot(fig1)

    st.subheader("🟢 Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Sentiment', data=df, palette='pink', ax=ax2)
    st.pyplot(fig2)

    st.subheader("☁️ Word Cloud: Positive Reviews")
    pos_reviews = ' '.join(df[df['Sentiment'] == 1]['Review Text'])
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white',
                              stopwords=STOPWORDS, colormap='pink').generate(pos_reviews)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud_pos, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

    st.subheader("☁️ Word Cloud: Negative Reviews")
    neg_reviews = ' '.join(df[df['Sentiment'] == 0]['Review Text'])
    wordcloud_neg = WordCloud(width=800, height=400, background_color='white',
                              stopwords=STOPWORDS, colormap='Purples').generate(neg_reviews)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.imshow(wordcloud_neg, interpolation='bilinear')
    ax4.axis('off')
    st.pyplot(fig4)

    st.subheader("✅ Confusion Matrix")
    X = df['Review Text']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    fig5, ax5 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax5)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    st.pyplot(fig5)
