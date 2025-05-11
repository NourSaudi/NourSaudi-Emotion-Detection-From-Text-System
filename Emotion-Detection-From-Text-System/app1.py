import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from googletrans import Translator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the pre-trained model and vectorizer
pipe_lr = joblib.load(open(r"C:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\streamlit-emotion-app\streamlit-emotion-app\support_vector_model.pkl", "rb"))
vectorizer = joblib.load(open(r"C:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\streamlit-emotion-app\streamlit-emotion-app\tfidf_vectorizer.pkl", "rb"))  # Load the saved TfidfVectorizer

# Dictionary mapping emotions to emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# Manually map integer predictions to emotion names based on the model's classes
emotion_mapping = {
    0: "anger",
    1: "fear",
    2: "joy"
    
}

# Function to predict emotions
def predict_emotions(docx):
    # Preprocess the input text
    vectorized_text = vectorizer.transform([docx])  # Convert text to numerical features
    results = pipe_lr.predict(vectorized_text)
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    # Preprocess the input text
    vectorized_text = vectorizer.transform([docx])  # Convert text to numerical features
    results = pipe_lr.predict_proba(vectorized_text)
    return results

# Function to highlight supported and unsupported words
def highlight_words(text, vectorizer_vocab):
    words = text.split()
    highlighted_text = ""
    for word in words:
        if word.lower() in vectorizer_vocab:
            # Highlight supported words in green
            highlighted_text += f"<span style='color: green;'>{word}</span> "
        else:
            # Highlight unsupported words in red
            highlighted_text += f"<span style='color: red;'>{word}</span> "
    return highlighted_text

# Main function for the Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text (Multi-Language Support)")

    # Initialize the translator
    translator = Translator()

    # Form for user input
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here (in any language)")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Translate the input text to English
        try:
            translated_text = translator.translate(raw_text, src='auto', dest='en').text
        except Exception as e:
            st.error("Translation failed. Please try again.")
            return

        # Get predictions and probabilities
        prediction = predict_emotions(translated_text)
        probability = get_prediction_proba(translated_text)

        # Convert prediction to string to match dictionary keys
        prediction_str = emotion_mapping[prediction]

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Translated Text")
            st.write(translated_text)

            # Highlight supported and unsupported words
            st.success("Word Highlighting")
            vectorizer_vocab = set(vectorizer.get_feature_names_out())
            highlighted_text = highlight_words(translated_text, vectorizer_vocab)
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction_str, "")
            st.write("{}:{}".format(prediction_str, emoji_icon))
            st.write("Confidence: {:.2f}%".format(np.max(probability) * 100))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=[emotion_mapping[i] for i in range(len(emotion_mapping))])
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            # Visualization using Plotly
            fig = px.bar(proba_df_clean, x='emotions', y='probability', color='emotions', title="Prediction Probability")
            st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == '__main__':
    main()