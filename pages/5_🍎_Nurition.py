import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# from pydataset import data
from streamlit_extras.no_default_selectbox import selectbox
import matplotlib.pyplot as plt

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

# Main Nutrition Calorie Tracker
df = pd.read_csv("./food1.csv", encoding='mac_roman')
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
        sel = selectbox('Select the food ', df['Shrt_Desc'].unique(), no_selection_label=" ", key=i)
        list1.append(sel)
        sel_serving = st.number_input('Select the number of servings ', min_value=1, max_value=10, value=1, step=1, key=j+100)
        i += 1
        j += 1
        st.write("Food : ", sel)
        st.write("Serving : ", sel_serving)
        st.write("Calories per serving : ", df[df['Shrt_Desc'] == sel]['Energ_Kcal'].values[0])
        cal = df[df['Shrt_Desc'] == sel]['Energ_Kcal'].values[0] * sel_serving
        list2.append(cal)
        st.write("Total calories for ", sel_serving, "servings of ", sel, "=", cal, "Energ_Kcal")

        # Protein
        protein = df[df['Shrt_Desc'] == sel]['Protein_(g)'].values[0] * sel_serving
        list3.append(protein)
        
        carbs = df[df['Shrt_Desc'] == sel]['Carbohydrt_(g)'].values[0] * sel_serving
        list4.append(carbs)
        
        fat = df[df['Shrt_Desc'] == sel]['Lipid_Tot_(g)'].values[0] * sel_serving
        list5.append(fat)
        
        sugar = df[df['Shrt_Desc'] == sel]['Sugar_Tot_(g)'].values[0] * sel_serving
        list7.append(sugar)
        
        calcium = df[df['Shrt_Desc'] == sel]['Calcium_(mg)'].values[0] * sel_serving
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

except:
    st.write("An error occurred. Please check your inputs.")
