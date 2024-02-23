
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load the pickled model
model_file = 'smote_pipeline_model.pkl'
with open(model_file, 'rb') as file:
    smote_pipeline = pickle.load(file)

# Streamlit app
def predict_sentiment(sentence):
    prediction = smote_pipeline.predict([sentence])
    return prediction[0]

def main():
    st.title("Sentiment Analysis App")

    # User input
    user_input = st.text_area("Enter a sentence:")
    if not user_input:
        st.warning("Please enter a sentence.")
        st.stop()

    # Make prediction
    prediction = predict_sentiment(user_input)

    # Display result
    st.subheader("Prediction:")
    st.write(prediction)

if __name__ == "__main__":
    main()
