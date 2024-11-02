import toml
import streamlit as st
import time
import requests

# Load the Hugging Face API key from a configuration file
with open("D:/GIT/AI-Fitness-Trainer/models/pages/secrets.toml", "r") as f:
    config = toml.load(f)

# Initialize Hugging Face API key
hf_api_key = config["huggingface"]["api_key"]

# Function to send a request to the Hugging Face Inference API
def get_hf_response(prompt):
    url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return "Sorry, I encountered an error."

# Function to display messages
def show_messages(text):
    messages_str = [
        f"<span style='color: green;'><b>USER</b>: {_['content']}</span></br>" if _['role'] == 'user' else f"<span style='color: white;'><b>SYSTEM</b>: {_['content']}</span></br></br>"
        for _ in st.session_state["messages"][1:]  # Skip the system message for display
    ]
    text.markdown("Messages", unsafe_allow_html=True)
    text.markdown(str("\n".join(messages_str)), unsafe_allow_html=True)

# Define the initial prompt for the assistant
BASE_PROMPT = [{"role": "system", "content": """
You are Donnie, an automated Gym assistant to provide workout routines for the users and give suggestions. \
You first greet the customer, then ask them what type of workout routine they want, \
give them a few workout options and wait for them to finalize. If they ask for changes, make those changes accordingly. \
Then summarize it and check for a final time if the user wants to add anything else. \
If it's a split, you ask for an upper body, lower body, or back chest and legs split. \
Make sure to clarify all questions about exercises and form. \
Also, make sure to talk only about fitness and fitness-related topics. \
You respond in a short, very conversational friendly style.
"""}]

# Initialize session state for storing messages if not already set
if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

# Streamlit app layout
st.header("FIT-BOT")
st.write("Start a conversation with the bot by typing in the box below. The bot will respond to your messages in a short, very conversational friendly style.")

# Placeholder for displaying chat messages
text = st.empty()
show_messages(text)

# Input state for the message box
if "something" not in st.session_state:
    st.session_state.something = ""

# Function to handle submission of user input
def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ""  # Clear input after submission

# Text input box for user messages
st.text_input("Enter message here", key="widget", on_change=submit)

# Process and generate response if there is new user input
if st.session_state.something != "":
    with st.spinner("Generating response..."):
        # Append user message to chat history
        st.session_state["messages"].append({"role": "user", "content": st.session_state.something})

        # Generate response from Hugging Face API
        prompt = "\n".join([msg["content"] for msg in st.session_state["messages"]])
        message_response = get_hf_response(prompt)

        # Append system response to chat history
        st.session_state["messages"].append({"role": "system", "content": message_response})

        # Update chat display
        show_messages(text)

    # Reset input state after processing
    st.session_state.something = ""

# Clear chat history when "Clear" button is pressed
if st.button("Clear"):
    st.session_state["messages"] = BASE_PROMPT
    show_messages(text)
