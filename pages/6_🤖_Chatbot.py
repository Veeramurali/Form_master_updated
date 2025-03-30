import openai
import toml
import streamlit as st

# Load configuration from the correct path of secrets.toml
with open("D:/GIT/AI-Fitness-Trainer/models/pages/secrets.toml", "r") as f:
    config = toml.load(f)

# Set OpenAI API key from secrets.toml
openai.api_key = config["openai"]["api_key"]

# Define base prompt
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

        **Meal Plan Categories**:
        1. **Veg Diet**:
            - Breakfast: The **Banana Oats Honey Seeds Milkshake** is mandatory with **30g protein**.
            - Divide the remaining calorie and protein needs across **Lunch** and **Dinner**.  
            - Keep **Dinner light** and nutrient-rich.
            - Add a **Snack** only if required to balance nutrient gaps.
        
        2. **Veg + Egg Diet**:
            - Breakfast: The **Banana Oats Honey Seeds Milkshake** with **30g protein**.
            - Include **eggs** in Lunch or Dinner for additional protein.
            - Suggest snacks with commonly available options like boiled eggs if necessary.

        3. **Non-Veg Diet**:
            - Breakfast: The **Banana Oats Honey Seeds Milkshake** with **30g protein**.
            - Include lean protein sources like **chicken**, **fish**, or **mutton** in Lunch and Dinner.
            - Suggest a **Snack** if additional calories or protein are needed.

        **Rules for Calculations**:
            - Ensure **calorie and protein breakdowns** for each meal are precise, with minimal deviations from the total requirement (± 10-20 kcal, ± 1-2g protein).
            - Double-check totals for accuracy before presenting the plan.

        **Workout Instructions**:
            - Provide **tailored workout plans** aligned with the user's fitness goals, including:
                - Warm-ups  
                - Targeted exercises for specific muscle groups  
                - Cool-down routines  
            - Ensure the workout plan includes **progressive overload** and **variety** to keep users engaged.

        **Output Example**:
        - Clearly specify meals with accurate measurements and their calorie/protein content.
        - Add preparation instructions for clarity.
        - Highlight adjustments to meet precise nutrition requirements, such as additional snacks or slight variations in quantities.
        """
    }
]



# Initialize session state if it's not already initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

# Function to display chat messages
def show_messages(text):
    messages_str = [
        f"<span style='color: green;'><b>USER</b>: {msg['content']}</span><br>" 
        if msg['role'] == 'user' 
        else f"<span style='color: white;'><b>SYSTEM</b>: {msg['content']}</span><br><br>"
        for msg in st.session_state["messages"][1:]  # Skip the initial system message
    ]
    text.markdown("Messages", unsafe_allow_html=True)
    text.markdown("".join(messages_str), unsafe_allow_html=True)

# Streamlit app UI
st.header("FIT-BOT")
st.write("Start a conversation with the bot by typing in the box below. The bot will respond in a friendly, conversational style.")

text_container = st.empty()  # For displaying chat messages
show_messages(text_container)

# Track the dynamic key for text input
if "input_key" not in st.session_state:
    st.session_state.input_key = "widget"  # Initial key for text input

# Text input for user message with dynamic key
user_input = st.text_input("Enter your message here:", key=st.session_state.input_key)

# Process the message when the input is submitted
if user_input:
    with st.spinner("Generating response..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        try:
            # Generate response from OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state["messages"]
            )
            bot_message = response["choices"][0]["message"]["content"]
            
            # Add bot message to chat history
            st.session_state["messages"].append({"role": "system", "content": bot_message})
            show_messages(text_container)

        except Exception as e:
            st.error(f"Error: {e}")
    
    # Reset the input by changing the key (this clears the widget)
    st.session_state.input_key = "widget_" + str(len(st.session_state["messages"]))  # Update key to reset widget

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state["messages"] = BASE_PROMPT
    st.session_state.input_key = "widget"  # Reset the input key to its initial state
    show_messages(text_container)
