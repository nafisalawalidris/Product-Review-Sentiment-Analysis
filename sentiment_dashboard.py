import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the trained model and vectorizer
model = joblib.load(r"C:\Users\USER\Downloads\Sentiment Analysis for Product Reviews\sentiment_model.pkl")
tfidf_vectorizer = joblib.load(r"C:\Users\USER\Downloads\Sentiment Analysis for Product Reviews\vectorizer.pkl")

# Title of the dashboard
st.title("🌟 Product Review Sentiment Analysis 🌟")

# Input section for user reviews
st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review:", "")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    if user_review:
        # Preprocess and vectorize the input review
        review_vector = tfidf_vectorizer.transform([user_review])
        
        # Predict sentiment
        prediction = model.predict(review_vector)
        
        # Map numerical prediction to sentiment label
        sentiment_map = {0: "Negative (1-2 stars)", 1: "Neutral (3 stars)", 2: "Positive (4-5 stars)"}
        predicted_sentiment = sentiment_map[prediction[0]]
        
        # Display the predicted sentiment
        st.success(f"The predicted sentiment is: **{predicted_sentiment}**")
    else:
        st.warning("Please enter a review before predicting.")

# Load sample data for visualization (replace with your actual data)
data = pd.DataFrame({
    'Sentiment': np.random.choice(['Negative', 'Neutral', 'Positive'], 100),
})

# Visualization of sentiment distribution
st.header("📊 Sentiment Distribution")
sentiment_count = data['Sentiment'].value_counts()

# Use seaborn for a more visually appealing bar plot
plt.figure(figsize=(10, 5))
sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='viridis')
plt.title("Distribution of Sentiments", fontsize=18, fontweight='bold')
plt.xlabel("Sentiment", fontsize=14)
plt.ylabel("Count", fontsize=14)
st.pyplot(plt)

# Word Cloud for the most frequent words in reviews
st.header("☁️ Word Cloud of Reviews")
wordcloud_data = ' '.join(data['Sentiment'])  # Replace with actual review text
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='plasma').generate(wordcloud_data)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
st.pyplot(plt)
