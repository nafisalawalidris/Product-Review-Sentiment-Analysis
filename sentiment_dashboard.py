import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model and vectorizer
model = joblib.load(r"C:\Users\USER\Downloads\Sentiment Analysis for Product Reviews\sentiment_model.pkl")
tfidf_vectorizer = joblib.load(r"C:\Users\USER\Downloads\Sentiment Analysis for Product Reviews\vectorizer.pkl")

# Function to predict sentiment
def predict_sentiment(review_text):
    review_vectorized = tfidf_vectorizer.transform([review_text])
    sentiment = model.predict(review_vectorized)
    return "positive" if sentiment[0] == 1 else "negative"

# Streamlit App UI
st.title("Sentiment Analysis for Product Reviews")

# Text input for user to enter a review
review_input = st.text_area("Enter a product review:", "")

if st.button("Predict Sentiment"):
    if review_input:
        # Predict the sentiment
        sentiment_result = predict_sentiment(review_input)
        
        # Display the result
        st.write(f"Sentiment of the review is **{sentiment_result}**")
    else:
        st.write("Please enter a review text.")

# Visualization Section
st.subheader("Sentiment Distribution")

# Example: Show a simple pie chart of sentiment distribution
# You can replace this with real data if available
labels = ['Positive', 'Negative']
sizes = [60, 40]  # Example distribution percentages
colors = ['#00FF00', '#FF0000']
explode = (0.1, 0)  # explode 1st slice (Positive)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

st.subheader("Sentiment Trends Over Time (Example)")

# Example trend over time (replace with actual data if available)
days = np.arange(1, 11)
positive_sentiments = np.random.randint(30, 70, size=10)
negative_sentiments = 100 - positive_sentiments

fig2, ax2 = plt.subplots()
ax2.plot(days, positive_sentiments, label='Positive Sentiment', marker='o', color='#00FF00')
ax2.plot(days, negative_sentiments, label='Negative Sentiment', marker='x', color='#FF0000')

ax2.set_xlabel('Day')
ax2.set_ylabel('Sentiment Percentage')
ax2.set_title('Sentiment Trends Over Time')
ax2.legend()

st.pyplot(fig2)

