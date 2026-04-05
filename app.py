import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# App title
st.title("📱 SMS Spam Detector")
st.write("Type any SMS message below and I'll tell you if it's Spam or Not.")

# Text input box
message = st.text_area("Enter your SMS message here:")

# Button
if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed = vectorizer.transform([message])
        result = model.predict(transformed)[0]
        if result == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is HAM — Not Spam!")