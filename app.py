import streamlit as st
from page1 import medical_image_enhancement
from page2 import medical_image_analysis
from page3 import medical_symptom_checker
from page4 import lab_report_explainer

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Medical Image Enhancement",
"Medical Image Analysis", "Symptom Checker", "Medical Interpreter"])
# page = st.sidebar.radio("Go to", ["Medical Image Analysis", "Symptom Checker", "Medical Interpreter"])

if page == "Medical Image Enhancement":
     medical_image_enhancement()
if page == "Medical Image Analysis":
    medical_image_analysis()
elif page == "Symptom Checker":
    medical_symptom_checker()
elif page == "Medical Interpreter":
    lab_report_explainer()






