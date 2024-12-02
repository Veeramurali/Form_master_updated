import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.no_default_selectbox import selectbox
import os

st.set_page_config(page_title='Nutrition Calorie Tracker', layout='wide')

html = """
<div style="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Nutrition Calorie Tracker</h2>
</div>"""
st.markdown(html, unsafe_allow_html=True) 

st.write("")

# BMI Calculator
st.subheader("BMI Calculator")
weight = st.number_input("Enter your weight (kg):", min_value=1.0, max_value=300.0, step=0.1)
height = st.number_input("Enter your height (cm):", min_value=50.0, max_value=250.0, step=0.1)
if weight and height:
    bmi = weight / ((height / 100) ** 2)
    st.write("Your BMI is:", round(bmi, 2))

# Calorie Requirement Calculator
st.subheader("Calorie Requirement Calculator")
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select your gender:", options=["Male", "Female"])
activity_level = st.selectbox("Select your activity level:", 
                              options=["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])

# Calculate BMR (Basic Metabolic Rate)
if gender == "Male":
    bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
else:
    bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

# Adjust BMR based on activity level
activity_multipliers = {
    "Sedentary": 1.2,
    "Lightly active": 1.375,
    "Moderately active": 1.55,
    "Very active": 1.725,
    "Extra active": 1.9
}
maintenance_calories = bmr * activity_multipliers[activity_level]
bulk_calories = maintenance_calories + 500  # Additional 500 calories for bulking
cut_calories = maintenance_calories - 500   # Reduce 500 calories for cutting

st.write("To maintain your weight, you need approximately:", round(maintenance_calories), "kcal per day.")
st.write("To gain weight, you need approximately:", round(bulk_calories), "kcal per day.")
st.write("To lose weight, you need approximately:", round(cut_calories), "kcal per day.")

# Protein Requirement Calculator
st.subheader("Protein Requirement Calculator")
goal = st.selectbox("What is your goal?", ["Maintain weight", "Bulk (gain weight)", "Cut (lose fat)"])

# Recommended protein intake per kg of body weight based on goal
protein_factors = {
    "Maintain weight": 1.0,
    "Bulk (gain weight)": 1.5,
    "Cut (lose fat)": 1.2
}
protein_needed = weight * protein_factors[goal]
st.write("Your recommended daily protein intake is:", round(protein_needed, 2), "grams.")

# Divider line before the main tracker
st.write("###")

# Diet selection: veg or non-veg
diet_type = st.selectbox("Select your diet type:", ["Veg", "Non-Veg"])

# Check if the selected file exists
veg_file_path = "D:/GIT/AI-Fitness-Trainer/models/pages/veg.xlsx"
non_veg_file_path = "D:/GIT/AI-Fitness-Trainer/models/pages/non_veg.xlsx"

if diet_type == "Veg" and os.path.exists(veg_file_path):
    df = pd.read_excel(veg_file_path)
elif diet_type == "Non-Veg" and os.path.exists(non_veg_file_path):
    df = pd.read_excel(non_veg_file_path)
else:
    st.error("File not found! Please check the path and try again.")
    st.stop()



ye = st.number_input('Enter Number of dishes', min_value=1, max_value=10)
i = 0
j = 0
calories = 0
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []

try:
    while(i < ye):
        st.write("--------------------")
        sel = selectbox('Select the food ', df['Food Item'].unique(), no_selection_label=" ", key=i)
        list1.append(sel)
        sel_serving = st.number_input('Select the number of servings ', min_value=1, max_value=10, value=1, step=1, key=j+100)
        i += 1
        j += 1
        st.write("Food : ", sel)
        st.write("Serving : ", sel_serving)
        
        # Check if food item exists in the dataframe
        if sel not in df['Food Item'].values:
            st.write(f"Error: {sel} is not found in the food list.")
            continue
        
        # Get food details safely
        food_data = df[df['Food Item'] == sel].iloc[0]
        st.write("Calories per serving : ", food_data['Calories (kcal)'])
        cal = food_data['Calories (kcal)'] * sel_serving
        list2.append(cal)
        st.write("Total calories for ", sel_serving, "servings of ", sel, "=", cal, "Energ_Kcal")

        # Protein per serving
        protein_per_serving = food_data['Protein (g)']
        protein = protein_per_serving * sel_serving
        list3.append(protein)
        st.write("Protein per serving : ", protein_per_serving, "g")
        st.write("Total protein for ", sel_serving, "servings of ", sel, "=", protein, "g")
        
        carbs = food_data['Carbohydrates (g)'] * sel_serving
        list4.append(carbs)
        
        fat = food_data['Fat (g)'] * sel_serving
        list5.append(fat)
        
        sugar = food_data['Sugar (g)'] * sel_serving
        list7.append(sugar)
        
        calcium = food_data['Calcium (mg)'] * sel_serving
        list8.append(calcium)
        
        calories += cal

    st.write("Total Calories:", calories)
    st.write("--------------------")

    col1, col2, col3 = st.columns(3)

    # Create pie charts
    with col1:
        fig = go.Figure(data=[go.Pie(labels=list1, values=list2, textinfo='percent', insidetextorientation='radial')])
        fig.update_layout(title="Calorie Breakdown")
        st.plotly_chart(fig)
    with col2:
        fig1 = go.Figure(data=[go.Pie(labels=list1, values=list3, textinfo='percent', insidetextorientation='radial')])
        fig1.update_layout(title="Protein Breakdown")
        st.plotly_chart(fig1)
    with col3:
        fig2 = go.Figure(data=[go.Pie(labels=list1, values=list4, textinfo='percent', insidetextorientation='radial')])
        fig2.update_layout(title="Carbs Breakdown")
        st.plotly_chart(fig2)
    with col1:
        fig3 = go.Figure(data=[go.Pie(labels=list1, values=list5, textinfo='percent', insidetextorientation='radial')])
        fig3.update_layout(title="Fat Breakdown")
        st.plotly_chart(fig3)
    with col3:
        fig5 = go.Figure(data=[go.Pie(labels=list1, values=list7, textinfo='percent', insidetextorientation='radial')])
        fig5.update_layout(title="Sugar Breakdown")
        st.plotly_chart(fig5)
    with col2:
        fig6 = go.Figure(data=[go.Pie(labels=list1, values=list8, textinfo='percent', insidetextorientation='radial')])
        fig6.update_layout(title="Calcium Breakdown")
        st.plotly_chart(fig6)

except Exception as e:
    st.write(f"An error occurred: {e}")

# Take inputs for designing a customized diet
st.write("###")
st.subheader("Customize Your Diet Plan")

# User inputs
target_calories = st.number_input("Enter your target calorie requirement (kcal):", min_value=00, max_value=5000, step=50)
target_protein = st.number_input("Enter your target protein requirement (g):", min_value=0, max_value=300, step=1)
diet_type = st.selectbox("Select your preferred diet plan:", ["Veg Diet", "Egg+Veg Diet", "Non-Veg Diet"])

veg_diet_plans = [
    {
        "plan": 1,
        "calories": 1800,
        "protein": 50,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 30,  # grams
                "Milk": 200,  # ml
                "Banana": 1,  # medium
                "Peanut Butter": 10  # grams
            },
            "Mid-Morning Snack": {
                "Apple": 1  # whole
            },
            "Lunch": {
                "Brown Rice": 150,  # grams cooked
                "Toor Dal": 150,  # grams cooked
                "Mixed Veggies": 100  # grams
            },
            "Evening Snack": {
                "Greek Yogurt": 100,  # grams
                "Pomegranate": 50  # grams
            },
            "Dinner": {
                "Roti": 2,  # whole wheat (60g each)
                "Paneer": 100,  # grams cooked
                "Broccoli": 100  # grams steamed
            }
        },
        "comments": "Add more protein-rich foods (like paneer or Greek yogurt) to increase protein intake."
    },
    {
        "plan": 2,
        "calories": 1900,
        "protein": 60,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 35,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 15
            },
            "Mid-Morning Snack": {
                "Orange": 1,
                "Roasted Chana": 30  # grams
            },
            "Lunch": {
                "Quinoa": 150,  # grams cooked
                "Rajma Curry": 150,  # grams
                "Cucumber Salad": 100  # grams
            },
            "Evening Snack": {
                "Moong Dal Chilla": 50,  # grams moong dal flour
                "Green Chutney": 1  # tbsp
            },
            "Dinner": {
                "Rice": 100,  # grams cooked
                "Veggies": 100,
                "Raita": 100  # ml
            }
        },
        "comments": "Increase protein content by adding a protein smoothie or extra beans in lunch."
    },
    {
        "plan": 3,
        "calories": 2000,
        "protein": 70,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 40,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 20
            },
            "Mid-Morning Snack": {
                "Idlis": 2,
                "Coconut Chutney": 1  # tsp
            },
            "Lunch": {
                "Chapati": 2,
                "Palak Paneer": 100,  # grams
                "Salad": 100  # grams
            },
            "Evening Snack": {
                "Fruit Chaat": 100  # grams
            },
            "Dinner": {
                "Millets": 150,  # grams cooked
                "Raita": 100  # ml
            }
        },
        "comments": "Increase protein by adding extra paneer or tofu."
    },
    {
        "plan": 4,
        "calories": 2100,
        "protein": 80,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 45,
                "Milk": 300,
                "Banana": 1,
                "Peanut Butter": 25
            },
            "Mid-Morning Snack": {
                "Sprouts Salad": 50,  # grams
                "Chopped Veggies": 50  # grams
            },
            "Lunch": {
                "Daliya": 100,  # grams cooked
                "Mixed Veggies": 100,
                "Curd": 100  # ml
            },
            "Evening Snack": {
                "Chana Sundal": 50,  # grams
                "Coconut": 10  # grams
            },
            "Dinner": {
                "Whole Wheat Flour": 60,  # grams
                "Veggies": 100  # grams
            }
        },
        "comments": "Increase protein by adding a larger portion of tofu or legumes."
    },
    {
        "plan": 5,
        "calories": 2200,
        "protein": 90,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 50,
                "Milk": 350,
                "Banana": 1,  # large
                "Peanut Butter": 30
            },
            "Mid-Morning Snack": {
                "Boiled Sweet Potato": 1  # medium
            },
            "Lunch": {
                "Brown Rice": 175,  # grams cooked
                "Chole": 200,  # grams
                "Cucumber-Tomato Salad": 100
            },
            "Evening Snack": {
                "Peanut Chutney Sandwich": 2  # slices
            },
            "Dinner": {
                "Whole Wheat Roti": 2,  # 60g each
                "Tofu": 100,  # grams
                "Vegetables": 100  # grams
            }
        },
        "comments": "Ensure protein target is met with larger servings of chole or tofu."
    },
    {
        "plan": 6,
        "calories": 2300,
        "protein": 100,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 50,
                "Milk": 400,
                "Banana": 1,  # large
                "Peanut Butter": 35
            },
            "Mid-Morning Snack": {
                "Papaya": 100  # grams
            },
            "Lunch": {
                "Quinoa": 175,  # grams cooked
                "Moong Dal": 150,  # grams cooked
                "Mixed Vegetables": 100
            },
            "Evening Snack": {
                "Protein Smoothie": 1  # scoop
            },
            "Dinner": {
                "Rice": 100,  # grams cooked
                "Lentils": 100,  # grams
                "Raita": 100  # ml
            }
        },
        "comments": "Consider adding extra protein powder to smoothie or add more legumes."
    },
    {
        "plan": 7,
        "calories": 2400,
        "protein": 110,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 55,
                "Milk": 450,
                "Banana": 1,  # large
                "Peanut Butter": 40
            },
            "Mid-Morning Snack": {
                "Boiled Chickpeas": 50  # grams
            },
            "Lunch": {
                "Vegetable Pulao": 150,  # grams cooked
                "Paneer": 100,  # grams
                "Mixed Salad": 100  # grams
            },
            "Evening Snack": {
                "Sprouted Moong Salad": 50  # grams
            },
            "Dinner": {
                "Chapati": 2,  # 60g each
                "Daal Tadka": 200,  # grams
                "Broccoli": 100  # grams steamed
            }
        },
        "comments": "Increase protein by adding more paneer or chickpeas."
    },
    {
        "plan": 8,
        "calories": 2500,
        "protein": 120,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 60,
                "Milk": 500,
                "Banana": 1,  # large
                "Peanut Butter": 50
            },
            "Mid-Morning Snack": {
                "Mixed Nuts": 30  # grams
            },
            "Lunch": {
                "Brown Rice": 200,  # grams cooked
                "Rajma": 150,  # grams
                "Spinach Curry": 100  # grams
            },
            "Evening Snack": {
                "Vegetable Sandwich": 2  # slices
            },
            "Dinner": {
                "Tofu": 150,  # grams
                "Millet": 100,  # grams cooked
                "Salad": 100  # grams
            }
        },
        "comments": "Add extra protein powder or tofu in dinner if protein is below target."
    },
    {
        "plan": 9,
        "calories": 2600,
        "protein": 130,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                "Oats": 65,
                "Milk": 500,
                "Banana": 1,  # large
                "Peanut Butter": 55
            },
            "Mid-Morning Snack": {
                "Greek Yogurt": 100,  # grams
                "Mango": 50  # grams
            },
            "Lunch": {
                "Quinoa": 200,  # grams cooked
                "Moong Dal": 200,  # grams cooked
                "Salad": 100  # grams
            },
            "Evening Snack": {
                "Fruit and Nut Bar": 1  # medium
            },
            "Dinner": {
                "Whole Wheat Roti": 3,  # 60g each
                "Lentil Soup": 200,  # grams
                "Kale": 100  # grams
            }
        },
        "comments": "Add larger portions of lentil soup or more yogurt to increase protein."
    },
    {
        "plan": 10,
        "calories": 2700,
        "protein": 140,
        "meals": {
            "Breakfast: Oats Banana Milkshake": {
                    "Oats": 70,  # grams
                    "Milk": 600,  # ml
                    "Banana": 1,  # large
                    "Peanut Butter": 70  # grams
                
            },
            "Mid-Morning Snack": {
                "Almonds": 30,  # grams
                "Greek Yogurt": 150  # grams
            },
            "Lunch": {
                "Brown Rice": 200,  # grams cooked
                "Tofu": 150,  # grams cooked
                "Mixed Veggies": 100  # grams
            },
            "Evening Snack": {
                "Protein Smoothie": 1,  # scoop
                "Cucumber Slices": 50  # grams
            },
            "Dinner": {
                "Quinoa": 150,  # grams cooked
                "Lentils": 150,  # grams cooked
                "Broccoli": 100,  # grams steamed
                "Whole Wheat Roti": 2  # 60g each
            }
        },
        "comments": "To meet your protein target, consider adding extra tofu or protein powder to the smoothie."
    }
]

veg_and_egg_diet = [
    {
        "plan": 1,
        "calories": 1800,
        "protein": 50,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 30,  # grams
                "Milk": 200,  # ml
                "Banana": 1,  # medium
                "Peanut Butter": 10,  # grams
                "Boiled Egg Whites": 3  # egg whites
            },
            "Mid-Morning Snack": {
                "Apple": 1,  # whole
                "Boiled Egg": 1  # whole egg
            },
            "Lunch": {
                "Brown Rice": 150,  # grams cooked
                "Toor Dal": 150,  # grams cooked
                "Mixed Veggies": 100,  # grams
                "Boiled Egg Whites": 2  # egg whites
            },
            "Evening Snack": {
                "Greek Yogurt": 100,  # grams
                "Pomegranate": 50,  # grams
            },
            "Dinner": {
                "Roti": 2,  # whole wheat (60g each)
                "Paneer": 100,  # grams cooked
                "Broccoli": 100,  # grams steamed
                "Boiled Egg Whites": 2  # egg whites
            }
        },
        "comments": "Add more protein-rich foods (like paneer or Greek yogurt) and include boiled eggs to meet protein intake."
    },
    {
        "plan": 2,
        "calories": 1900,
        "protein": 60,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 35,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 15,
                "Boiled Egg Whites": 3  # egg whites
            },
            "Mid-Morning Snack": {
                "Orange": 1,
                "Roasted Chana": 30,  # grams
            },
            "Lunch": {
                "Quinoa": 150,  # grams cooked
                "Rajma Curry": 150,  # grams
                "Cucumber Salad": 100,  # grams
            },
            "Evening Snack": {
                "Moong Dal Chilla": 50,  # grams moong dal flour
                "Green Chutney": 1,  # tbsp
                "Boiled Egg": 1  # whole egg
            },
            "Dinner": {
                "Rice": 100,  # grams cooked
                "Veggies": 100,
                "Raita": 100,  # ml
                "Boiled Egg Whites": 2  # egg whites
            }
        },
        "comments": "Increase protein content by adding a protein smoothie or extra beans in lunch, along with eggs."
    },
    {
        "plan": 3,
        "calories": 2000,
        "protein": 70,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 40,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 20,
                "Boiled Egg Whites": 4  # egg whites
            },
            "Mid-Morning Snack": {
                "Idlis": 2,
                "Coconut Chutney": 1,  # tsp
            },
            "Lunch": {
                "Chapati": 2,
                "Palak Paneer": 100,  # grams
                "Salad": 100,  # grams
            },
            "Evening Snack": {
                "Fruit Chaat": 100,  # grams
            },
            "Dinner": {
                "Millets": 150,  # grams cooked
                "Raita": 100,  # ml
            }
        },
        "comments": "Increase protein by adding extra paneer or tofu, along with boiled eggs."
    },
    {
        "plan": 4,
        "calories": 2100,
        "protein": 80,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 45,
                "Milk": 300,
                "Banana": 1,
                "Peanut Butter": 25,
                "Boiled Egg Whites": 5  # egg whites
            },
            "Mid-Morning Snack": {
                "Sprouts Salad": 50,  # grams
                "Chopped Veggies": 50,  # grams
            },
            "Lunch": {
                "Daliya": 100,  # grams cooked
                "Mixed Veggies": 100,
                "Curd": 100,  # ml
            },
            "Evening Snack": {
                "Chana Sundal": 50,  # grams
                "Coconut": 10,  # grams
            },
            "Dinner": {
                "Whole Wheat Flour": 60,  # grams
                "Veggies": 100,  # grams
            }
        },
        "comments": "Increase protein by adding a larger portion of tofu or legumes, along with boiled eggs."
    },
    {
        "plan": 5,
        "calories": 2200,
        "protein": 90,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 50,
                "Milk": 350,
                "Banana": 1,  # large
                "Peanut Butter": 30,
                "Boiled Egg Whites": 5  # egg whites
            },
            "Mid-Morning Snack": {
                "Boiled Sweet Potato": 1,  # medium
            },
            "Lunch": {
                "Brown Rice": 175,  # grams cooked
                "Chole": 200,  # grams
                "Cucumber-Tomato Salad": 100,
            },
            "Evening Snack": {
                "Peanut Chutney Sandwich": 2,  # slices
            },
            "Dinner": {
                "Whole Wheat Roti": 2,  # 60g each
                "Tofu": 100,  # grams
                "Vegetables": 100,  # grams
                "Boiled Egg Whites": 2  # egg whites
            }
        },
        "comments": "Ensure protein target is met with larger servings of chole or tofu, and include boiled eggs."
    },
    {
        "plan": 6,
        "calories": 2300,
        "protein": 100,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 50,
                "Milk": 400,
                "Banana": 1,  # large
                "Peanut Butter": 35,
                "Boiled Egg Whites": 6  # egg whites
            },
            "Mid-Morning Snack": {
                "Papaya": 100,  # grams
            },
            "Lunch": {
                "Quinoa": 175,  # grams cooked
                "Moong Dal": 150,  # grams cooked
                "Mixed Vegetables": 100,
            },
            "Evening Snack": {
                "Protein Smoothie": 1,  # scoop
            },
            "Dinner": {
                "Rice": 100,  # grams cooked
                "Lentils": 100,  # grams
                "Raita": 100,  # ml
            }
        },
        "comments": "Consider adding extra protein powder to smoothie or add more legumes, and add boiled eggs."
    },
    {
        "plan": 7,
        "calories": 2400,
        "protein": 110,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 55,
                "Milk": 450,
                "Banana": 1,  # large
                "Peanut Butter": 40,
                "Boiled Egg Whites": 6  # egg whites
            },
            "Mid-Morning Snack": {
                "Mixed Fruit Salad": 150,  # grams
            },
            "Lunch": {
                "Brown Rice": 200,  # grams cooked
                "Palak Dal": 200,  # grams cooked
                "Cucumber Salad": 100,
            },
            "Evening Snack": {
                "Roasted Chana": 50,  # grams
            },
            "Dinner": {
                "Whole Wheat Roti": 2,
                "Grilled Vegetables": 100,  # grams
                "Boiled Egg Whites": 3
            }
        },
        "comments": "Add protein-dense foods such as tofu or dals, and incorporate eggs to meet the target."
    },
    {
        "plan": 8,
        "calories": 2500,
        "protein": 120,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 60,
                "Milk": 500,
                "Banana": 1,  # large
                "Peanut Butter": 45,
                "Boiled Egg Whites": 6  # egg whites
            },
            "Mid-Morning Snack": {
                "Apple": 1,
                "Roasted Peanuts": 30,
            },
            "Lunch": {
                "Quinoa": 200,  # grams cooked
                "Rajma": 150,  # grams cooked
                "Cucumber Salad": 100,
            },
            "Evening Snack": {
                "Sprouts Salad": 100,
            },
            "Dinner": {
                "Chapati": 2,
                "Vegetables": 150,  # grams
                "Boiled Egg Whites": 3
            }
        },
        "comments": "Add extra legumes or protein-based smoothies and eggs to boost protein."
    },
    {
        "plan": 9,
        "calories": 2600,
        "protein": 130,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 65,
                "Milk": 550,
                "Banana": 1,
                "Peanut Butter": 50,
                "Boiled Egg Whites": 7
            },
            "Mid-Morning Snack": {
                "Boiled Sweet Potato": 1,  # medium
                "Chana Sundal": 50,  # grams
            },
            "Lunch": {
                "Millets": 200,  # grams cooked
                "Lentils": 150,  # grams cooked
                "Veggies": 150,  # grams
            },
            "Evening Snack": {
                "Protein Bar": 1,
            },
            "Dinner": {
                "Chapati": 3, 
                "Tofu": 100,
                "Salad": 100,
                "Boiled Egg Whites": 4
            }
        },
        "comments": "Increase protein-rich food, including legumes, tofu, or extra protein powder and boiled eggs."
    },
    {
        "plan": 10,
        "calories": 2700,
        "protein": 140,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 70,
                "Milk": 600,
                "Banana": 1, 
                "Peanut Butter": 55,
                "Boiled Egg Whites": 8
            },
            "Mid-Morning Snack": {
                "Mixed Fruit Salad": 150, 
                "Almonds": 10,
            },
            "Lunch": {
                "Brown Rice": 250,
                "Lentils": 200,
                "Cucumber-Tomato Salad": 150,
            },
            "Evening Snack": {
                "Protein Smoothie": 1,
            },
            "Dinner": {
                "Whole Wheat Roti": 3,
                "Vegetables": 200, 
                "Boiled Egg Whites": 3
            }
        },
        "comments": "Ensure proper protein intake by increasing protein-rich foods and adding more boiled eggs."
    }
]

non_veg_diet = [
    {
        "plan": 1,
        "calories": 1800,
        "protein": 50,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 30,  # grams
                "Milk": 200,  # ml
                "Banana": 1,  # medium
                "Peanut Butter": 10,  # grams
                "Boiled Egg Whites": 3  # egg whites
            },
            "Mid-Morning Snack": {
                "Apple": 1,  # whole
            },
            "Lunch: Grilled Chicken Salad": {
                "Chicken Breast": 100,  # grams
                "Mixed Veggies": 100,  # grams
                "Olive Oil": 1,  # tsp
                "Lemon Juice": 1,  # tbsp
            },
            "Evening Snack": {
                "Greek Yogurt": 100,  # grams
                "Pomegranate": 50,  # grams
            },
            "Dinner: Veggie Stir Fry with Eggs": {
                "Mixed Veggies": 100,  # grams
                "Boiled Egg Whites": 2  # egg whites
            }
        },
        "comments": "This plan provides a good balance with a variety of non-veg options and a focus on egg whites."
    },
    {
        "plan": 2,
        "calories": 1900,
        "protein": 60,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 35,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 15,
                "Boiled Egg Whites": 3  # egg whites
            },
            "Mid-Morning Snack": {
                "Orange": 1,
                "Roasted Chana": 30,  # grams
            },
            "Lunch: Grilled Chicken with Brown Rice": {
                "Chicken Breast": 150,  # grams
                "Brown Rice": 150,  # grams cooked
                "Cucumber Salad": 50,  # grams
            },
            "Evening Snack": {
                "Boiled Egg": 1,  # whole egg
                "Fruit Chaat": 100,  # grams
            },
            "Dinner: Paneer Tikka with Veggies": {
                "Paneer": 100,  # grams
                "Mixed Vegetables": 100,  # grams
                "Roti": 2  # whole wheat
            }
        },
        "comments": "The plan balances non-veg (chicken) and vegetarian options for a varied protein intake."
    },
    {
        "plan": 3,
        "calories": 2000,
        "protein": 70,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 40,
                "Milk": 250,
                "Banana": 1,
                "Peanut Butter": 20,
                "Boiled Egg Whites": 4  # egg whites
            },
            "Mid-Morning Snack": {
                "Idlis": 2,
                "Coconut Chutney": 1,  # tsp
            },
            "Lunch: Grilled Chicken and Quinoa": {
                "Chicken Breast": 150,  # grams
                "Quinoa": 150,  # grams cooked
                "Mixed Veggies": 100,  # grams
            },
            "Evening Snack": {
                "Greek Yogurt": 100,  # grams
                "Roasted Almonds": 20,  # grams
            },
            "Dinner: Veggie Soup with Boiled Egg": {
                "Mixed Vegetables": 100,  # grams
                "Boiled Egg Whites": 2  # egg whites
            }
        },
        "comments": "Chicken and quinoa provide ample protein, while eggs and veggies keep it balanced."
    },
    {
        "plan": 4,
        "calories": 2100,
        "protein": 80,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 45,
                "Milk": 300,
                "Banana": 1,
                "Peanut Butter": 25,
                "Boiled Egg Whites": 5  # egg whites
            },
            "Mid-Morning Snack": {
                "Orange": 1,
                "Boiled Egg": 1  # whole egg
            },
            "Lunch: Grilled Salmon with Brown Rice": {
                "Salmon": 150,  # grams
                "Brown Rice": 150,  # grams cooked
                "Mixed Veggies": 100,  # grams
            },
            "Evening Snack": {
                "Fruit Salad": 100,  # grams
            },
            "Dinner: Chicken Stir Fry with Veggies": {
                "Chicken Breast": 100,  # grams
                "Mixed Vegetables": 100,  # grams
            }
        },
        "comments": "The combination of grilled salmon and chicken ensures a rich non-veg protein source."
    },
    {
        "plan": 5,
        "calories": 2200,
        "protein": 90,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 50,
                "Milk": 350,
                "Banana": 1,  # large
                "Peanut Butter": 30,
                "Boiled Egg Whites": 6  # egg whites
            },
            "Mid-Morning Snack": {
                "Boiled Sweet Potato": 1,  # medium
            },
            "Lunch: Grilled Chicken and Palak (Spinach)": {
                "Chicken Breast": 150,  # grams
                "Palak": 100,  # grams cooked
                "Roti": 2,  # whole wheat
            },
            "Evening Snack": {
                "Moong Dal Chilla": 50,  # grams moong dal flour
                "Green Chutney": 1,  # tbsp
            },
            "Dinner: Paneer Tikka with Mixed Vegetables": {
                "Paneer": 100,  # grams
                "Mixed Vegetables": 100,  # grams
                "Cucumber Salad": 50,  # grams
            }
        },
        "comments": "This plan balances grilled chicken and vegetarian meals while providing ample protein."
    },
    {
        "plan": 6,
        "calories": 2300,
        "protein": 100,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 50,
                "Milk": 400,
                "Banana": 1,  # large
                "Peanut Butter": 35,
                "Boiled Egg Whites": 6  # egg whites
            },
            "Mid-Morning Snack": {
                "Papaya": 100,  # grams
            },
            "Lunch: Grilled Chicken with Veggies": {
                "Chicken Breast": 150,  # grams
                "Mixed Vegetables": 150,  # grams
                "Quinoa": 150,  # grams cooked
            },
            "Evening Snack": {
                "Protein Smoothie": 1,  # scoop
            },
            "Dinner: Paneer and Veggie Stir Fry": {
                "Paneer": 100,  # grams
                "Mixed Vegetables": 100,  # grams
            }
        },
        "comments": "With a strong focus on protein through chicken and paneer, this plan maintains variety."
    },
    {
        "plan": 7,
        "calories": 2400,
        "protein": 110,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 55,
                "Milk": 450,
                "Banana": 1,  # large
                "Peanut Butter": 40,
                "Boiled Egg Whites": 7  # egg whites
            },
            "Mid-Morning Snack": {
                "Mixed Fruit Salad": 150,  # grams
            },
            "Lunch: Grilled Fish and Brown Rice": {
                "Fish (Salmon)": 150,  # grams
                "Brown Rice": 200,  # grams cooked
                "Cucumber-Tomato Salad": 100,  # grams
            },
            "Evening Snack": {
                "Roasted Chana": 50,  # grams
            },
            "Dinner: Chicken and Vegetable Stew": {
                "Chicken Breast": 100,  # grams
                "Mixed Vegetables": 100,  # grams
            }
        },
        "comments": "Grilled fish and chicken provide ample protein while maintaining meal variety."
    },
    {
        "plan": 8,
        "calories": 2500,
        "protein": 120,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 60,
                "Milk": 500,
                "Banana": 1,  # large
                "Peanut Butter": 45,
                "Boiled Egg Whites": 8  # egg whites
            },
            "Mid-Morning Snack": {
                "Cucumber Salad": 100,  # grams
            },
            "Lunch: Grilled Chicken with Spinach": {
                "Chicken Breast": 200,  # grams
                "Spinach": 100,  # grams cooked
                "Quinoa": 150,  # grams cooked
            },
            "Evening Snack": {
                "Greek Yogurt": 100,  # grams
                "Almonds": 30,  # grams
            },
            "Dinner: Tofu Stir Fry with Vegetables": {
                "Tofu": 150,  # grams
                "Mixed Vegetables": 100,  # grams
                "Whole Wheat Roti": 2  # whole wheat
            }
        },
        "comments": "This plan contains a combination of non-veg and vegetarian meals to keep protein intake high."
    },
    {
        "plan": 9,
        "calories": 2600,
        "protein": 130,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 65,
                "Milk": 550,
                "Banana": 1,
                "Peanut Butter": 50,
                "Boiled Egg Whites": 7  # egg whites
            },
            "Mid-Morning Snack": {
                "Boiled Sweet Potato": 1,  # medium
                "Chana Sundal": 50,  # grams
            },
            "Lunch: Millets and Chicken": {
                "Millets": 200,  # grams cooked
                "Chicken Breast": 150,  # grams
                "Mixed Veggies": 100,  # grams
            },
            "Evening Snack": {
                "Protein Bar": 1,
            },
            "Dinner: Chicken Tikka with Veggies": {
                "Chicken": 200,  # grams
                "Mixed Vegetables": 100,  # grams
            }
        },
        "comments": "Chicken adds a great source of protein while ensuring a balanced diet."
    },
    {
        "plan": 10,
        "calories": 2700,
        "protein": 140,
        "meals": {
            "Breakfast: Oats Banana Milkshake with Eggs": {
                "Oats": 70,
                "Milk": 600,
                "Banana": 1, 
                "Peanut Butter": 55,
                "Boiled Egg Whites": 8  # egg whites
            },
            "Mid-Morning Snack": {
                "Mixed Fruit Salad": 150, 
                "Almonds": 10,
            },
            "Lunch: Brown Rice and Chicken": {
                "Brown Rice": 250,
                "Chicken": 200,  # grams
                "Cucumber-Tomato Salad": 150,
            },
            "Evening Snack": {
                "Protein Smoothie": 1,
            },
            "Dinner: Whole Wheat Roti with Chicken": {
                "Whole Wheat Roti": 3,
                "Chicken": 100,  # grams
                "Mixed Vegetables": 200, 
            }
        },
        "comments": "Ensure proper protein intake through chicken and balanced vegetarian meals."
    }
]


def get_best_diet_plan(target_calories, target_protein, diet_plans):
    best_plan = None
    smallest_diff = float('inf')  # Initialize to a large number to find the smallest difference

    for plan in diet_plans:
        calorie_diff = abs(plan['calories'] - target_calories)
        protein_diff = abs(plan['protein'] - target_protein)
        total_diff = calorie_diff + protein_diff  # You can also apply weighting here if needed

        if total_diff < smallest_diff:
            smallest_diff = total_diff
            best_plan = plan

    return best_plan

if target_calories > 0 and target_protein > 0:
    st.write("Your personalized diet plans will be shown below:")

    # Choose the appropriate diet plan list based on user input
    if diet_type == "Veg Diet":
        best_plan = get_best_diet_plan(target_calories, target_protein, veg_diet_plans)
    elif diet_type == "Egg+Veg Diet":
        best_plan = get_best_diet_plan(target_calories, target_protein, veg_and_egg_diet)
    elif diet_type == "Non-Veg Diet":
        best_plan = get_best_diet_plan(target_calories, target_protein, non_veg_diet)

    # Display the best diet plan
    if best_plan:
        st.write(f"Based on your target calories and protein, we recommend Diet Plan {best_plan['plan']}:")
        st.write(f"Calories: {best_plan['calories']} kcal, Protein: {best_plan['protein']} g")
        st.write("### Meals:")

        for meal, items in best_plan['meals'].items():
            st.write(f"**{meal}:**")
            for food_item, quantity in items.items():
                st.write(f" - {food_item}: {quantity} grams")

        st.write("### Comment:")
        st.write(best_plan["comments"])
    else:
        st.write("No suitable diet plan found.")
else:
    st.write("Please enter valid calorie and protein requirements to see your diet plans.")