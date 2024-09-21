import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
# If you saved your vectorizer, load it too
# vectorizer = joblib.load('vectorizer.pkl')

# Sample data for demonstration (replace this with your actual data)
data = pd.DataFrame({
    'review_headline': ['Great product!', 'Not what I expected.', 'Loved it!', 'Terrible service.'],
    'review_body': ['I love it.', 'It was okay.', 'Best purchase ever!', 'I will not buy again.'],
    'sentiment': [2, 0, 2, 0]  # Use 2 for positive, 0 for negative, etc.
})

# Function to predict sentiment
def predict_sentiment(review_headline, review_body):
    input_data = pd.DataFrame([[review_headline + ' ' + review_body]], columns=['text'])
    # Vectorization can be done here if needed
    # X_vectorized = vectorizer.transform(input_data)
    sentiment = model.predict(input_data)[0]
    sentiment_label = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_label[sentiment]

# Streamlit UI
st.title("Sentiment Analysis Dashboard")
st.write("Analyze customer reviews and visualize sentiment.")

# Input for user review
st.subheader("Input Your Review")
review_headline = st.text_input("Review Headline")
review_body = st.text_area("Review Body")
if st.button("Predict Sentiment"):
    sentiment = predict_sentiment(review_headline, review_body)
    st.success(f"The predicted sentiment is: {sentiment}")

# Display sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_counts = data['sentiment'].value_counts()
sentiment_counts.index = sentiment_counts.index.map({0: "negative", 1: "neutral", 2: "positive"})

# Bar chart
plt.figure(figsize=(10, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'gray', 'green'])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution")
st.pyplot(plt)

# Provide insights
st.subheader("Business Insights and Recommendations")
st.write("""
- **Negative Feedback**: Highlight pain points from negative reviews and consider addressing them in product updates or customer support strategies.
- **Positive Feedback**: Identify strengths and leverage them in marketing campaigns.
- **Neutral Reviews**: Look for opportunities to convert neutral feedback into positive by enhancing customer experience.
""")

# Run the app
if __name__ == "__main__":
    st.run()