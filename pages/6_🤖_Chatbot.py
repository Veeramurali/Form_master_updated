import streamlit as st
from transformers import pipeline
import spacy

# Load spaCy model for NLP tasks like tokenization and lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy model 'en_core_web_sm' using the command: python -m spacy download en_core_web_sm")
    st.stop()

# Load Hugging Face transformer model for generating responses
generator = pipeline('conversational', model='microsoft/DialoGPT-medium')

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Function to display chat history
def show_messages():
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f"<span style='color: green;'><b>USER:</b> {message['content']}</span><br>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: white;'><b>BOT:</b> {message['content']}</span><br>", unsafe_allow_html=True)

# Function to handle user input
def submit():
    user_input = st.session_state.input_box
    if user_input:
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Generate response using the transformer model
        bot_response = generator(user_input)[0]['generated_text']
        st.session_state["messages"].append({"role": "bot", "content": bot_response})

        # Clear the input box
        st.session_state.input_box = ""

# Title and description
st.header("Chat with FitBot")
st.write("Type a message and the bot will respond.")

# Display conversation history
show_messages()

# Text input for user messages
st.text_input("Your message:", key="input_box", on_change=submit)

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.experimental_rerun()
