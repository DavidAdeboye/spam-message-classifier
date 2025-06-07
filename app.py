import streamlit as st
import pickle

#load the model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
#check if the model loaded successfully
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("✅ Model loaded successfully!")

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("Spam Classifier")
st.write("This app classifies messages as spam or ham (not spam).")

#text input for user message
user_input = st.text_area("Enter a message:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        prediction = model.predict([user_input])
        label = "Spam" if prediction[0] == 1 else "Not Spam"
        st.subheader("Prediction")
        st.success(f"The message is classified as: **{label}**")