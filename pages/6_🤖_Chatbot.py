import streamlit as st
import requests

# ✅ Cohere API endpoint and your API key
API_URL = "https://api.cohere.ai/v1/chat"
API_KEY = "your api key"

# Base prompt for the assistant
BASE_PROMPT = [
    {
        "role": "system",
        "content": """
        You are Donnie, an automated Gym assistant designed to provide users with highly accurate workout plans and diet plans tailored to their fitness goals.

        **For diet-related queries**:
            - Provide a **detailed diet plan** tailored to the user's **calorie** and **protein** requirements.
            - The diet plan should consist of **Breakfast**, **Lunch**, **Dinner**, and an optional **Snack** to balance calorie or protein requirements if necessary.
            - **Breakfast** must always include a **Banana Oats Honey Seeds Milkshake** that provides **30 grams of protein** and an appropriate calorie count. Adjust the recipe to ensure precise values.
            - For other meals:
                - Prioritize **accuracy** in calorie and protein calculations for all food items and the total diet plan.  
                - Ensure the food choices are **practical, nutrient-dense, and budget-friendly**.  
                - Incorporate a balanced mix of **protein**, **carbs**, and **healthy fats** in each meal.  
                - Use widely available and easy-to-prepare ingredients.
            - Provide **specific quantities** for each item in **grams or commonly used measures**.
            - Include **meal preparation recommendations** for every meal to ensure clarity for the user.
            - At the end, provide a **total calorie and protein breakdown** of the entire plan and specify adjustments for additional or missing nutrients.
        """
    }
]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

# Function to show the chat history
def show_messages(container):
    formatted_messages = []
    for msg in st.session_state["messages"][1:]:  # Skip system message
        if msg["role"] == "user":
            formatted_messages.append(f"<span style='color: green;'><b>USER</b>: {msg['content']}</span><br>")
        else:
            formatted_messages.append(f"<span style='color: white;'><b>DONNIE</b>: {msg['content']}</span><br><br>")
    container.markdown("".join(formatted_messages), unsafe_allow_html=True)

# Streamlit UI
st.header("FIT-BOT")
st.write("Start a conversation with Donnie by typing below:")

text_container = st.empty()
show_messages(text_container)

if "input_key" not in st.session_state:
    st.session_state.input_key = "widget"

user_input = st.text_input("Enter your message here:", key=st.session_state.input_key)

if user_input:
    with st.spinner("Generating response..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})

        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            history = [
                {"role": msg["role"], "message": msg["content"]}
                for msg in st.session_state["messages"]
            ]

            data = {
                "chat_history": history,
                "message": user_input
            }

            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            reply = response.json()["text"]

            st.session_state["messages"].append({"role": "system", "content": reply})
            show_messages(text_container)

        except Exception as e:
            st.error(f"Error: {e}")

    st.session_state.input_key = "widget_" + str(len(st.session_state["messages"]))

# Clear chat button
if st.button("Clear Conversation"):
    st.session_state["messages"] = BASE_PROMPT
    st.session_state.input_key = "widget"
    show_messages(text_container)
